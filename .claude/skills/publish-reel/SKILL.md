---
name: publish-reel
description: Prepare and publish a rendered turntable film (renders/*.mp4 from this repo's CUDA renderer) as an Instagram Reel through the Graph API (Facebook Login, resumable upload - no public URL) - caption with the model's required attribution, music under the film, the user's approval of the exact file, then the post. Use when the user asks to prepare for publishing / publish / post / wrzuć / opublikuj a render or Reel on Instagram, to add music to a film, or to check the Instagram account or token.
---

# Preparing and publishing a Reel

Taken over from StlViewer's `publish-reel` skill (2026-11-02). `publish.mjs` (next to this file)
talks to the Graph API (`graph.facebook.com`, Facebook Login for Business); the film is uploaded
straight from disk to `rupload.facebook.com` (resumable upload), so nothing is hosted publicly.
Instagram Login tokens can't do that (Meta docs: resumable upload is "only for apps that have
implemented Facebook Login for Business"). `add-music.mjs` puts a track under the film.

The film comes from the `render-turntable` skill: `renders/<name>.mp4`, H.264 9:16 1080x1920 at
~9-10 Mbit/s with `-Drt.turntable.quality=max`, no sound - fine for Reels as it is.

## Account and token

- **@kingmidasmodels** (IG id 17841465538673583), Facebook Page "Midas renders", Meta app
  "Midas renders publisher" (id 2841776309530529) - the same account StlViewer posts on (the
  user's decision, 2026-11-02).
- The script reads StlViewer's `~/.config/stlviewer/instagram.json` (`IG_CONFIG=<file>` points
  elsewhere). Its Page token was made long-lived on 2026-10-29 and does not expire. Use the file
  where it is; never copy it or the token anywhere.
- The token **never** goes into the repo, a commit, a skill, memory or a web form.

## Steps

1. `node .claude/skills/publish-reel/publish.mjs check` - account, quota (100 posts / 24 h),
   token expiry. A token that expires: see "Token" below.
2. **Caption** - write it to `renders/<name>.txt` (git-ignored, next to the film):
   - one hook line in English (e.g. "The Midas touch: Yennefer of Vengerberg, turned to gold."),
   - **the model's attribution line** - required by CC-BY / CC-BY-ND licences. Models come from
     StlViewer's model repository: `C:/Users/szymondomeracki/Documents/projects/StlViewer/models-to-render/models/NNN_*/INFO.md`,
     line "Uznanie autorstwa (wymagane): ..." - the repository is a working directory of this
     session, so read it straight from there, and check its "Licencja – co wolno" section: an
     NC, NC-ND or personal-use model is not for a Reel. Yennefer's:
     "The Witcher - Yennefer" by LM3Dtisk Lukáš Mlýnek, CC-BY-ND, https://www.printables.com/model/1673033-the-witcher-yennefer
   - **no line on how it was made** - the user's decision (2026-11-02): no renderer, tool or
     site name in the caption (StlViewer's "Rendered in the browser with simplestlviewer.com"
     does not apply here),
   - a few hashtags (#3dprinting #render #raytracing #fanart #<character> ...).
3. **Music** (optional, automatic) - see "Music" below; the result is `renders/<name>-music.mp4`.
4. **Approval - always, for every Reel, no exceptions.** The user's standing rule: a film
   prepared for publishing must be approved before it is published. Send the user (SendUserFile,
   so it reaches their phone) **the exact file that will be published** - the final MP4 with the
   music already in it, not the silent render - plus the cover frame (extract it with
   `ffmpeg -ss <s> -i <film> -frames:v 1 <cover.png>`), and show the full caption and the audio
   name. Then wait for an explicit "tak"/yes to *that* version **from the user**. Any change after
   the approval (other music fragment, caption, cover, re-render) needs a new approval.
   Publishing is public and can't be quietly undone; never publish on your own initiative, and
   an earlier "publish X" - from the user or from a managing session such as rt-orchestrator - is
   not approval of a film the user hasn't seen.
5. `node .claude/skills/publish-reel/publish.mjs reel renders/<name>-music.mp4 renders/<name>.txt`
   (`--cover-ms=N` picks the cover frame at N ms, `--no-feed` keeps it off the grid,
   `--audio-name="Midas - <Character>"` names the Reel's own audio - it can be set only once).
   There is no dry run: it publishes as soon as Instagram has processed the upload. It prints the
   container id, the upload, the processing status every 15 s and the permalink.
6. Report the permalink.

## Music

The public Graph API has no field for Instagram's licensed music library (only `audio_name`,
the name of the Reel's own audio; library access is a partner deal). Two ways:

1. **Track in the file** (automatic):
   `node .claude/skills/publish-reel/add-music.mjs renders/<film>.mp4 <track> [renders/<film>-music.mp4] [--start=S|auto] [--fade-out=S]`
   - cuts the track to the film, fades it, -14 LUFS, video copied untouched; `--start=auto`
   takes the loudest stretch. Publish that file with `--audio-name=...`; it becomes the Reel's
   original audio. Only tracks the user may use commercially (Pixabay Music, YouTube Audio
   Library "no attribution", own music; CC-BY tracks need their credit in the caption too).
   Tracks are in `renders/music/` (git-ignored): 15 Pixabay Music tracks in folders by theme -
   `gry/` (games, e.g. The Witcher), `komiksy/` (comics), `filmy/` (films), `anime/`, `groza/`
   (horror) - copied from StlViewer on 2026-11-02 at the user's request, with
   `renders/music/SOURCES.md` listing each track's page and the Pixabay Content License
   (commercial use, no attribution needed, never pass the track on as a file of its own).
   Pick from the folder that fits the character and tell the user which track and fragment.
   A new track goes into `SOURCES.md` with its source and licence; **never download one
   without the user's OK on the file and the source**, and never put someone else's music in a
   file.
2. **Trending audio** (manual, best reach): send the user the film and the caption; they post it
   in the Instagram app and pick music there.

## Token

`setup <token>` (or `IG_TOKEN=... setup`) stores the token in the config file. From a user token
it takes the Page's token. A Page token made from a **long-lived** user token never expires;
from a short-lived one (Graph API Explorer gives ~1-2 h) it dies with it. To get a long-lived
one either the user pastes the Explorer token into
https://developers.facebook.com/tools/debug/accesstoken/ → **Extend Access Token** (desktop only -
the tools pages don't open on a phone), or, from a phone, gives you the App secret (App settings
→ Basic) and you exchange the short token yourself:
`GET /oauth/access_token?grant_type=fb_exchange_token&client_id=<app id>&client_secret=<secret>&fb_exchange_token=<token>`,
then `setup` with the result. Never store the secret; afterwards tell the user to **Reset** it
(the Page token stays valid). You may not type tokens into web forms, and the Chrome extension
won't read them off a page. Needed scopes: `instagram_basic`, `instagram_content_publish`,
`pages_read_engagement`, `pages_show_list`.

## Requirements and notes

- node with built-in `fetch` (ESM `.mjs`, top-level await); ffmpeg + ffprobe on the PATH (scoop)
  for `add-music.mjs`; internet to `graph.facebook.com` and `rupload.facebook.com`.
- Licences: the attribution line is mandatory for CC-BY / CC-BY-ND models. Yennefer's INFO.md
  allows commercial use with attribution and forbids publishing modified files (a render is not
  the STL file); the character is someone else's IP (fan art), so monetising it is a legal risk
  whatever the licence. What to publish is the user's call.
- Reels specs: MP4 H.264, 9:16; our ~9-10 Mbit/s 1080x1920 films are fine.
