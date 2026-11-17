#!/usr/bin/env node
// Zernio (zernio.com) posting API for the films of @kingmidasmodels: YouTube (public post) and TikTok
// (Creator Inbox DRAFT only). Instagram goes through ../publish-reel/publish.mjs (Graph API).
// Taken over unchanged from StlViewer's publish-film skill (2026-11-17): same account, same key file.
// API key: ~/.config/stlviewer/zernio.json {"api_key": "sk_..."} - never print, copy or commit it.
//
//   node .claude/skills/publish-film/zernio.mjs accounts
//   node .claude/skills/publish-film/zernio.mjs youtube renders/<film>-music.mp4 --caption=renders/<film>-music.txt
//        [--title="..."] [--thumbnail=renders/<cover>.jpg] [--visibility=public|unlisted|private]
//        [--category=24] [--tags=a,b] [--dry-run]
//   node .claude/skills/publish-film/zernio.mjs tiktok-draft renders/<film>.mp4 [--caption=renders/<film>.txt] [--dry-run]
//   node .claude/skills/publish-film/zernio.mjs status POST_ID
//   node .claude/skills/publish-film/zernio.mjs wait POST_ID [--minutes=N]
//
// Where the text goes (docs.zernio.com/platforms/youtube/posts, /platforms/tiktok/posts, checked 2026-11-03):
// - YouTube: title = platforms[].platformSpecificData.title (max 100; the top-level `title` is NOT used),
//   description = `content` (max 5000, hashtags written in it show on the video), keywords = top-level
//   `tags` (invisible, from the caption's hashtags). The top-level `hashtags` field is stored only.
// - TikTok draft: the video goes to the creator's inbox; TikTok ignores every post_info field on drafts,
//   so caption, cover and privacy are set in the TikTok app. `content` is sent only as Zernio's record.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { execFileSync } from 'node:child_process';

const API = 'https://zernio.com/api/v1';
const KEY_FILE = path.join(os.homedir(), '.config', 'stlviewer', 'zernio.json');
const MAX_HASHTAGS = 5;   // the user's rule for every platform (and Instagram's own limit)

class Fatal extends Error {}
const die = (msg) => { process.exitCode = 1; throw new Fatal(msg); };

function apiKey() {
  let cfg = null;
  try { cfg = JSON.parse(fs.readFileSync(KEY_FILE, 'utf8')); } catch { /* reported below */ }
  if (!cfg?.api_key) die(`No API key in ${KEY_FILE} ({"api_key": "sk_..."} from zernio.com -> API keys)`);
  return cfg.api_key;
}

async function api(method, p, body) {
  const res = await fetch(`${API}${p}`, {
    method,
    headers: { Authorization: `Bearer ${apiKey()}`, ...(body ? { 'Content-Type': 'application/json' } : {}) },
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  let json;
  try { json = JSON.parse(text || '{}'); } catch { json = { raw: text.slice(0, 500) }; }
  if (!res.ok) {
    const msg = JSON.stringify(json);
    if (/pending drafts/i.test(msg)) die('TikTok already holds 5 pending drafts for this account. Waiting does not free a slot: the user has to post or delete drafts in the TikTok app first.');
    die(`${method} ${p}: HTTP ${res.status} ${msg.slice(0, 1000)}`);
  }
  return [res.status, json ?? {}];
}

// ---- local checks -------------------------------------------------------------------------------

function ffprobeBin() {
  for (const b of ['ffprobe', path.join(os.homedir(), 'scoop', 'shims', 'ffprobe.exe')]) {
    try { execFileSync(b, ['-version'], { stdio: 'ignore' }); return b; } catch { /* next */ }
  }
  return null;
}

/** {width, height, duration, audio} of a video, or null without ffprobe. */
function probe(file) {
  const bin = ffprobeBin();
  if (!bin) return null;
  const j = JSON.parse(execFileSync(bin, ['-v', 'error', '-show_entries', 'stream=codec_type,width,height:format=duration', '-of', 'json', file], { encoding: 'utf8' }));
  const v = (j.streams || []).find((s) => s.codec_type === 'video') || {};
  return { width: v.width, height: v.height, duration: Number(j.format?.duration), audio: (j.streams || []).some((s) => s.codec_type === 'audio') };
}

function videoFile(file) {
  if (!file || !fs.existsSync(file) || !fs.statSync(file).isFile()) die(`File not found: ${file}`);
  const ctype = { '.mp4': 'video/mp4', '.mov': 'video/quicktime', '.webm': 'video/webm' }[path.extname(file).toLowerCase()];
  if (!ctype) die('Supported formats: .mp4, .mov, .webm');
  return ctype;
}

const hashtagsOf = (text) => [...text.matchAll(/#([\p{L}\p{N}_]+)/gu)].map((m) => m[1]);

function readCaption(file, required) {
  if (!file) { if (required) die('--caption=FILE.txt is required'); return ''; }
  if (!fs.existsSync(file)) die(`Caption file not found: ${file}`);
  const text = fs.readFileSync(file, 'utf8').replace(/\r\n/g, '\n').trim();
  const tags = hashtagsOf(text);
  if (tags.length > MAX_HASHTAGS) die(`${file} has ${tags.length} hashtags (${tags.map((t) => '#' + t).join(' ')}); the limit is ${MAX_HASHTAGS}. Fix the caption file first.`);
  return text;
}

// ---- Zernio -------------------------------------------------------------------------------------

async function account(platform, username) {
  const [, j] = await api('GET', `/accounts?platform=${platform}`);
  const list = (j.accounts || []).filter((a) => a.platform === platform);
  const acc = username ? list.find((a) => a.username === username || a.username === username.replace(/^@/, '')) : list[0];
  if (!acc) die(username ? `No ${platform} account @${username} connected in Zernio` : `No ${platform} account connected in Zernio`);
  if (!acc.isActive || acc.needsReconnection) die(`${platform} @${acc.username} needs reconnecting in Zernio`);
  return acc;
}

/** Presigned upload of a local file to Zernio's storage; returns its public URL. */
async function upload(file, ctype) {
  const size = fs.statSync(file).size;
  console.log(`Uploading ${path.basename(file)} (${(size / 1048576).toFixed(1)} MB) to Zernio`);
  const [, pre] = await api('POST', '/media/presign', { filename: path.basename(file), contentType: ctype, size });
  const put = await fetch(pre.uploadUrl, { method: 'PUT', headers: { 'Content-Type': ctype }, body: fs.readFileSync(file) });
  if (!put.ok) die(`Upload of ${file} failed: HTTP ${put.status} ${(await put.text()).slice(0, 300)}`);
  return pre.publicUrl;
}

const isDraftEntry = (p) =>
  Boolean(p.platformSpecificData?.isDraft || p.platformSpecificData?.tiktokSettings?.draft || String(p.platformPostId || '').startsWith('v_inbox'));

function printPost(post, status) {
  console.log(`Post ${post._id}: ${post.status}${status ? ` (HTTP ${status})` : ''}`);
  for (const p of post.platforms || []) {
    const extra = [
      p.platform === 'tiktok' && isDraftEntry(p) ? `DRAFT in the TikTok inbox (${p.platformPostId || 'no id yet'})` : '',
      p.platformPostUrl ? `url ${p.platformPostUrl}` : '',
      p.errorMessage || p.error ? `error: ${p.errorMessage || JSON.stringify(p.error)}` : '',
    ].filter(Boolean).join(', ');
    console.log(`  ${p.platform}: ${p.status}${extra ? ` - ${extra}` : ''}`);
  }
}

/** Prints the result; a 207 is still res.ok, so failure is read from post.status. */
function finish(j, status) {
  const post = j.post || {};
  printPost(post, status);
  for (const w of j.warnings || []) console.log(`  warning: ${typeof w === 'string' ? w : JSON.stringify(w)}`);
  if (['failed', 'partial', 'cancelled'].includes(post.status)) die(`Zernio reports the post as ${post.status}${j.message ? `: ${j.message}` : ''}`);
  return post;
}

// ---- commands -----------------------------------------------------------------------------------

async function cmdAccounts() {
  const [, j] = await api('GET', '/accounts');
  if (!(j.accounts || []).length) console.log('No accounts connected in Zernio.');
  for (const a of j.accounts || []) {
    console.log(`${a.platform}  @${a.username}  (${a.displayName})  id ${a._id}  ${a.isActive ? 'active' : 'inactive'}${a.needsReconnection ? ', needs reconnection' : ''}`);
  }
}

async function cmdYoutube(file, f) {
  const ctype = videoFile(file);
  const description = readCaption(f.caption, true);
  const title = (f.title || description.split('\n').find((l) => l.trim()) || '').trim();
  if (!title) die('No title: pass --title="..." or start the caption with a hook line');
  if (title.length > 100) die(`Title is ${title.length} characters, YouTube allows 100: pass a shorter --title`);
  if (/[<>]/.test(title + description)) die('YouTube does not allow < or > in the title or description');
  if (description.length > 5000) die(`Description is ${description.length} characters, YouTube allows 5000`);
  const visibility = f.visibility || 'public';
  if (!['public', 'unlisted', 'private'].includes(visibility)) die('--visibility must be public, unlisted or private');
  const tags = [...new Set([...hashtagsOf(description), ...(f.tags ? f.tags.split(',') : [])].map((t) => t.trim()).filter(Boolean))];
  if (f.thumbnail && !fs.existsSync(f.thumbnail)) die(`Thumbnail not found: ${f.thumbnail}`);
  if (f.thumbnail && fs.statSync(f.thumbnail).size > 2 * 1048576) die('YouTube thumbnails are limited to 2 MB');
  const thumbType = f.thumbnail && ({ '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png' }[path.extname(f.thumbnail).toLowerCase()] || die('Thumbnail must be .jpg or .png'));

  const info = probe(file);
  const short = info && info.height >= info.width && info.duration <= 180;
  const acc = await account('youtube', f.account);
  console.log(`Account:     YouTube @${acc.username} (${acc.displayName})`);
  console.log(`Video:       ${path.basename(file)}${info ? ` ${info.width}x${info.height}, ${info.duration.toFixed(1)} s, ${info.audio ? 'with sound' : 'NO SOUND'} -> ${short ? 'a Short' : 'a regular video (not vertical or over 3 min)'}` : ''}`);
  console.log(`Title:       ${title}`);
  console.log(`Visibility:  ${visibility}, category ${f.category || '24'}, made for kids: no, synthetic media: no`);
  console.log(`Tags:        ${tags.join(', ') || '(none)'}  (keywords, not shown on the video)`);
  console.log(`Thumbnail:   ${f.thumbnail || '(YouTube picks a frame)'}`);
  console.log(`Description (shown under the video, hashtags included):\n${description.replace(/^/gm, '  | ')}`);
  if (info && !info.audio) console.log('WARNING: the film has no sound; YouTube gets the music version.');
  if (f.dryRun) { console.log('\n--dry-run: nothing uploaded or posted.'); return; }

  const url = await upload(file, ctype);
  const thumb = f.thumbnail ? await upload(f.thumbnail, thumbType) : null;
  const [status, j] = await api('POST', '/posts', {
    content: description,
    tags,
    mediaItems: [{ type: 'video', url, ...(thumb ? { thumbnail: thumb } : {}) }],
    platforms: [{
      platform: 'youtube',
      accountId: acc._id,
      platformSpecificData: { title, visibility, categoryId: String(f.category || '24'), madeForKids: false, containsSyntheticMedia: false },
    }],
    publishNow: true,
  });
  const post = finish(j, status);
  const yt = (post.platforms || []).find((p) => p.platform === 'youtube');
  if (yt?.platformPostUrl) console.log(`YouTube: ${yt.platformPostUrl}`);
  else console.log(`No URL yet: node .claude/skills/publish-film/zernio.mjs wait ${post._id}`);
}

async function cmdTiktokDraft(file, f) {
  const ctype = videoFile(file);
  const caption = readCaption(f.caption, false);
  const info = probe(file);
  const acc = await account('tiktok', f.account);
  console.log(`Account:  TikTok @${acc.username} (${acc.displayName}) - as a DRAFT to the Creator Inbox, never published`);
  console.log(`Video:    ${path.basename(file)}${info ? ` ${info.width}x${info.height}, ${info.duration.toFixed(1)} s, ${info.audio ? 'WITH SOUND (the silent render is the TikTok one; the sound is picked in the app)' : 'silent'}` : ''}`);
  console.log('Caption:  TikTok ignores captions, covers and privacy on drafts; the user pastes the caption in the app.');
  if (f.dryRun) { console.log('\n--dry-run: nothing uploaded or posted.'); return; }

  const url = await upload(file, ctype);
  // draft: true at the root AND on the entry (the entry's value wins); there is no flag that turns it off.
  const draft = { draft: true, content_preview_confirmed: true, express_consent_given: true };
  const [status, j] = await api('POST', '/posts', {
    ...(caption ? { content: caption } : {}),
    mediaItems: [{ type: 'video', url }],
    platforms: [{ platform: 'tiktok', accountId: acc._id, platformSpecificData: { tiktokSettings: draft } }],
    tiktokSettings: draft,
    publishNow: true,   // hands the draft to TikTok now; Zernio's own isDraft would keep it in Zernio only
  });
  const post = finish(j, status);
  const tk = (post.platforms || []).find((p) => p.platform === 'tiktok');
  if (tk && tk.status === 'published' && !isDraftEntry(tk)) {
    die(`The TikTok entry of post ${post._id} carries no draft marker - check the profile NOW; if it went public, delete it there.`);
  }
  if (tk?.status === 'published') console.log('Draft handed to TikTok: the user opens the inbox notification in the TikTok app (31.8+), adds sound, caption and posts.');
  else console.log(`Not delivered yet: node .claude/skills/publish-film/zernio.mjs wait ${post._id}`);
}

async function cmdStatus(id) {
  if (!id) die('usage: status POST_ID');
  const [, j] = await api('GET', `/posts/${encodeURIComponent(id)}`);
  printPost(j.post || j);
}

// Polls every 20 s until the post leaves publishing (exit 0 published / 1 failed / 2 timeout).
async function cmdWait(id, minutes = 30) {
  if (!id) die('usage: wait POST_ID [--minutes=N]');
  const t0 = Date.now();
  let last = '';
  for (;;) {
    const [, j] = await api('GET', `/posts/${encodeURIComponent(id)}`);
    const post = j.post || j;
    const line = JSON.stringify([post.status, (post.platforms || []).map((p) => [p.status, p.platformPostUrl || null, p.errorMessage || p.error || null])]);
    if (line !== last) {
      console.log(`[${new Date().toLocaleTimeString('pl-PL')}]`);
      printPost(post);
      last = line;
    }
    if (['published', 'failed', 'partial', 'cancelled'].includes(post.status)) {
      if (post.status !== 'published') process.exitCode = 1;
      return;
    }
    if (Date.now() - t0 > minutes * 60000) {
      console.log(`TIMEOUT after ${minutes} min, still ${post.status}`);
      process.exitCode = 2;
      return;
    }
    await new Promise((r) => setTimeout(r, 20000));
  }
}

const argv = process.argv.slice(2);
const cmd = argv.shift();
const flags = {};
const pos = [];
for (const a of argv) {
  const m = /^--([a-z-]+)(?:=(.*))?$/s.exec(a);
  if (!m) { pos.push(a); continue; }
  const key = m[1].replace(/-([a-z])/g, (_, c) => c.toUpperCase());
  flags[key] = m[2] ?? true;
}
const cmds = {
  accounts: () => cmdAccounts(),
  youtube: () => cmdYoutube(pos[0], flags),
  'tiktok-draft': () => cmdTiktokDraft(pos[0], flags),
  draft: () => cmdTiktokDraft(pos[0], flags),   // old name
  status: () => cmdStatus(pos[0]),
  wait: () => cmdWait(pos[0], flags.minutes ? Number(flags.minutes) : undefined),
};
try {
  if (!cmds[cmd]) die('usage: zernio.mjs accounts | youtube FILE --caption=FILE.txt [--title=..] [--thumbnail=IMG] [--visibility=..] [--category=N] [--tags=a,b] [--dry-run] | tiktok-draft FILE [--caption=FILE.txt] [--dry-run] | status POST_ID | wait POST_ID [--minutes=N]');
  await cmds[cmd]();
} catch (e) {
  if (!(e instanceof Fatal)) throw e;
  console.error(`ERROR: ${e.message}`);
}
