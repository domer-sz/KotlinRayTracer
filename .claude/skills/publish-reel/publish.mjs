// Publishes a local MP4 as an Instagram Reel through the Graph API with Facebook Login
// (resumable upload: the file goes straight to rupload.facebook.com, no public URL needed).
// Taken over from StlViewer's publish-reel skill; only the config location differs.
//
//   node publish.mjs setup <token>          save the token, find the Instagram account, print it
//   node publish.mjs check                  account, token validity, publishing quota (read-only)
//   node publish.mjs reel <file.mp4> <caption.txt> [--cover-ms=N] [--no-feed] [--audio-name=Name]
//                                           create the container, upload, wait, publish
//
// The token and account live in StlViewer's ~/.config/stlviewer/instagram.json (never in the
// repo): the user decided both projects post on the same account. IG_CONFIG=<file> points
// elsewhere.
// Needs a token with instagram_basic, instagram_content_publish, pages_read_engagement,
// pages_show_list (Graph API Explorer, user or Page token of the Page linked to the account).
import { readFileSync, writeFileSync, mkdirSync, statSync } from 'node:fs';
import { homedir } from 'node:os';
import { join, dirname } from 'node:path';

const API = 'v26.0';
const GRAPH = `https://graph.facebook.com/${API}`;
const CONFIG = process.env.IG_CONFIG || join(homedir(), '.config', 'stlviewer', 'instagram.json');
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function load() {
  try { return JSON.parse(readFileSync(CONFIG, 'utf8')); } catch { return {}; }
}
function save(cfg) {
  mkdirSync(dirname(CONFIG), { recursive: true });
  writeFileSync(CONFIG, JSON.stringify(cfg, null, 2), { mode: 0o600 });
}

async function graph(method, path, params = {}, token) {
  const url = new URL(GRAPH + path);
  const body = new URLSearchParams({ ...params, access_token: token });
  const res = method === 'GET'
    ? await fetch(url + '?' + body)
    : await fetch(url, { method, body });
  const json = await res.json().catch(() => ({}));
  if (!res.ok || json.error) {
    const e = json.error || {};
    throw new Error(`${method} ${path}: ${e.message || res.status} (code ${e.code}, subcode ${e.error_subcode || '-'})`);
  }
  return json;
}

/**
 * Finds the Instagram professional account linked to one of the token's Pages. From a user
 * token it takes that Page's own token: made from a long-lived user token, it never expires.
 */
async function findAccount(token) {
  try {
    const me = await graph('GET', '/me', { fields: 'id,name,instagram_business_account{id,username}' }, token);
    if (me.instagram_business_account) return { igId: me.instagram_business_account.id, username: me.instagram_business_account.username, page: me.name, token };
  } catch { /* a user token: look through its Pages */ }
  const pages = await graph('GET', '/me/accounts', { fields: 'name,access_token,instagram_business_account{id,username}' }, token);
  const page = (pages.data || []).find((p) => p.instagram_business_account);
  if (!page) throw new Error('no Facebook Page with a linked Instagram professional account for this token');
  return { igId: page.instagram_business_account.id, username: page.instagram_business_account.username, page: page.name, token: page.access_token || token };
}

/** When the token stops working ("never" for a Page token from a long-lived user token). */
async function expiry(token) {
  const { data: d = {} } = await graph('GET', '/debug_token', { input_token: token }, token);
  return { type: d.type, expires: d.expires_at ? new Date(d.expires_at * 1000).toISOString() : 'never' };
}

async function check(cfg) {
  const acc = await graph('GET', `/${cfg.igId}`, { fields: 'username,name,media_count,followers_count' }, cfg.token);
  const quota = await graph('GET', `/${cfg.igId}/content_publishing_limit`, { fields: 'config,quota_usage' }, cfg.token);
  const q = (quota.data || [])[0] || {};
  console.log(`config ${CONFIG}`);
  console.log(`account @${acc.username} (${acc.id}), posts ${acc.media_count}, followers ${acc.followers_count}`);
  console.log(`published in the last 24 h: ${q.quota_usage ?? '?'} of ${(q.config && q.config.quota_total) ?? '?'}`);
  const e = await expiry(cfg.token);
  console.log(`token: ${e.type}, expires ${e.expires}`);
}

async function reel(cfg, file, captionFile, flags) {
  const size = statSync(file).size;
  const caption = readFileSync(captionFile, 'utf8').trim();
  const params = { media_type: 'REELS', upload_type: 'resumable', caption, share_to_feed: String(!flags.has('--no-feed')) };
  const cover = [...flags].find((f) => f.startsWith('--cover-ms='));
  if (cover) params.thumb_offset = cover.split('=')[1];
  const audioName = [...flags].find((f) => f.startsWith('--audio-name='));
  if (audioName) params.audio_name = audioName.slice('--audio-name='.length);   // can be set only once

  const { id } = await graph('POST', `/${cfg.igId}/media`, params, cfg.token);
  console.log('container', id);

  const up = await fetch(`https://rupload.facebook.com/ig-api-upload/${API}/${id}`, {
    method: 'POST',
    headers: { Authorization: `OAuth ${cfg.token}`, offset: '0', file_size: String(size) },
    body: readFileSync(file),
  });
  const upJson = await up.json().catch(() => ({}));
  if (!up.ok || !upJson.success) throw new Error('upload failed: ' + JSON.stringify(upJson));
  console.log(`uploaded ${(size / 1048576).toFixed(1)} MB`);

  for (let i = 0; ; i++) {
    const { status_code: s, status } = await graph('GET', `/${id}`, { fields: 'status_code,status' }, cfg.token);
    console.log('status', s);
    if (s === 'FINISHED') break;
    if (s === 'ERROR' || s === 'EXPIRED') throw new Error('processing failed: ' + (status || s));
    if (i >= 40) throw new Error('still processing after 10 min; container ' + id + ' can be published later');
    await sleep(15000);
  }

  const { id: mediaId } = await graph('POST', `/${cfg.igId}/media_publish`, { creation_id: id }, cfg.token);
  const media = await graph('GET', `/${mediaId}`, { fields: 'permalink,media_product_type' }, cfg.token);
  console.log('published', mediaId, media.media_product_type, media.permalink);
}

const [cmd, ...args] = process.argv.slice(2);
const cfg = load();
try {
  if (cmd === 'setup') {
    const token = args[0] || process.env.IG_TOKEN;
    if (!token) throw new Error('usage: setup <token>');
    const acc = await findAccount(token);
    save({ token: acc.token, igId: acc.igId, username: acc.username, page: acc.page, saved: new Date().toISOString() });
    console.log(`saved: @${acc.username} (${acc.igId}), Page "${acc.page}" -> ${CONFIG}`);
    await check(load());
  } else if (cmd === 'check') {
    if (!cfg.token) throw new Error(`no token yet in ${CONFIG}: run setup (or set IG_CONFIG)`);
    await check(cfg);
  } else if (cmd === 'reel') {
    if (!cfg.token) throw new Error(`no token yet in ${CONFIG}: run setup (or set IG_CONFIG)`);
    const [file, captionFile, ...rest] = args;
    if (!file || !captionFile) throw new Error('usage: reel <file.mp4> <caption.txt>');
    await reel(cfg, file, captionFile, new Set(rest));
  } else {
    console.log('usage: setup <token> | check | reel <file.mp4> <caption.txt> [--cover-ms=N] [--no-feed] [--audio-name=Name]');
  }
} catch (err) {
  console.error('error:', err.message);
  process.exit(1);
}
