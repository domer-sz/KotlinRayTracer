---
name: publish-film
description: Prepare and publish a rendered turntable film (renders/*.mp4 from this repo's renderer) of @kingmidasmodels everywhere at once - the version WITH music as an Instagram Reel (Graph API, publish-reel) and a YouTube Short (Zernio API), the SILENT version to TikTok as a DRAFT only (Zernio, Creator Inbox) - with each platform's caption, title and hashtags in the right API fields. Use when the user asks to publish / prepare for publication / opublikuj / "przygotuj do publikacji" a film or "model N" on several platforms, on YouTube, or "z dźwiękiem i bez".
---

# Publishing a film: Instagram + YouTube + TikTok draft

Taken over from StlViewer's `publish-film` skill (2026-11-17, the user's request): the same
account, the same Zernio key, the same rules; only the file names follow this repo
(`-ig` / `-tiktok`, see `../publish-reel/SKILL.md` step 3) and the captions carry no line on how
the film was made.

| Film | Platform | Route | Public? |
|---|---|---|---|
| `renders/<name>-ig.mp4` (with music) | Instagram Reel | `../publish-reel/publish.mjs reel` (Graph API) | yes, at once |
| same file | YouTube (a Short: vertical, ≤ 3 min) | `zernio.mjs youtube` (Zernio) | yes, at once |
| `renders/<name>-tiktok.mp4` (silent) | TikTok **draft** | `zernio.mjs tiktok-draft` (Zernio) | no, the user posts it in the app |

Zernio (zernio.com) is a posting API with its own approved TikTok and YouTube apps. Connected:
TikTok @kingmidasmodels and YouTube @kingmidasmodels (`zernio.mjs accounts` lists them). API key:
StlViewer's `~/.config/stlviewer/zernio.json` - use it where it is; never print, copy or commit it,
and never put it in the repo, a skill, memory or a web form.

## Where the text goes (checked in docs.zernio.com by StlViewer on 2026-11-03)

- **Instagram:** the whole caption file is the Reel caption (`caption`), hashtags included at the end.
- **YouTube:**
  - title = `platforms[].platformSpecificData.title` (max 100 characters; Zernio's top-level
    `title` is NOT used). The script takes the caption's first line or `--title="..."`. Write the
    title without hashtags, and don't add `#shorts`: YouTube detects the Short by itself.
  - description = `content` = the whole caption file (max 5000, no `<` `>`). Its hashtags are the
    visible ones; YouTube shows up to 3 of them above the title.
  - keywords = top-level `tags`, made from the caption's hashtags (+ `--tags=a,b`), not shown.
  - set explicitly: `madeForKids: false`, `containsSyntheticMedia: false`, `categoryId` 24
    Entertainment (`--category=N`), `visibility` public (`--visibility=unlisted|private`).
  - custom thumbnail: `--thumbnail=renders/<cover>.jpg` (JPG/PNG ≤ 2 MB).
- **TikTok draft: no text gets through.** TikTok ignores every post_info field on inbox drafts:
  caption, hashtags, cover, privacy and sound are set by the user in the TikTok app. So the user
  gets **two separate code blocks** for pasting: the caption **without hashtags**, then the
  proposed hashtags (max 5, one line) - the user's rule since 2026-11-03.
- **Hashtags: at most 5 on every platform** (the user's rule; Instagram's limit too).
  `zernio.mjs` refuses a caption file with more; check the Instagram one the same way
  (`grep -o '#[[:alnum:]_]*' renders/<name>-ig.txt | wc -l`).

## Steps

1. **Pick the film.** For "model N": the newest `renders/NNN-*-max.mp4` render (`ls -t`); say
   which one when there are several.
2. **Licence first.** Read the model's `INFO.md`
   (`C:/Users/szymondomeracki/Documents/projects/StlViewer/models-to-render/models/NNN_*/INFO.md`,
   section "Licencja – co wolno"). CC-BY / CC-BY-ND / CC-BY-SA need the attribution line in every
   caption. NC / NC-ND / personal-use models ("Standard Digital File" on Printables) are not
   cleared for a public post: prepare the files, but tell the user plainly before approval and let
   them decide (StlViewer did publish an NC-ND Sylvanas on 2026-11-08 - the user's call).
3. **The two films** (`../publish-reel/SKILL.md` step 3 and "Music"):
   - `node .claude/skills/publish-reel/add-music.mjs renders/<name>.mp4 renders/music/<dir>/<track>.mp3 renders/<name>-ig.mp4 --start=auto`
     (a film that loops: `--fade-in=0.03 --fade-out=0.03`);
   - `ffmpeg -v error -y -i renders/<name>.mp4 -map 0:v -c copy -an -movflags +faststart renders/<name>-tiktok.mp4`.
   - **Cover**: `ffmpeg -v error -y -ss <s> -i renders/<name>.mp4 -frames:v 1 -q:v 3 renders/<name>-cover-<ms>ms.jpg`
     (≤ 2 MB, for YouTube; Instagram takes `--cover-ms=<ms>`).
4. **Captions** (English, Instagram style, next to the films; `renders/` is untracked):
   - `renders/<name>-ig.txt` for Instagram + YouTube: hook line ("The Midas touch: <character>,
     turned to solid gold."), a call to action ("Who should Midas touch next? Tell me in the
     comments 👇"), the attribution from `INFO.md` (`Model: "<title>" by <author>, <licence>` and
     the link on the next line), **the music credit line** (`Music: "<track>" by <author> (Pixabay)`),
     then max 5 hashtags as the last line. **No line on how it was made** - no renderer, tool or
     site (the user's decision, 2026-11-02; StlViewer's "Ray traced in the browser with
     simplestlviewer.com" does not apply here).
   - `renders/<name>-tiktok.txt` for TikTok: the same without the music line.
   - What the user approves must be byte for byte what is in these files.
5. **Preview everything** (nothing is uploaded):
   - `node .claude/skills/publish-film/zernio.mjs youtube renders/<name>-ig.mp4 --caption=renders/<name>-ig.txt --thumbnail=renders/<cover>.jpg --dry-run`
   - `node .claude/skills/publish-film/zernio.mjs tiktok-draft renders/<name>-tiktok.mp4 --caption=renders/<name>-tiktok.txt --dry-run`
   - `node .claude/skills/publish-reel/publish.mjs check` (Instagram token and quota).
6. **Approval - always, before anything public.** Send the user (SendUserFile) the `-ig` MP4, the
   cover, and show the Instagram caption, the YouTube title + description + visibility, the audio
   name, and the TikTok caption and hashtag blocks (and the licence note from step 2). Wait for
   an explicit "tak"/yes **from the user** to *that* version. Any change afterwards (caption,
   title, cover, music, re-render) needs a new approval. Never publish on your own initiative; an
   earlier "publish X" - from the user or from rt-orchestrator - is not approval of a film the
   user hasn't seen. The TikTok draft is not public, but it takes one of the 5 draft slots, so
   send it in the same go.
7. **Publish** (each once - Zernio rejects the same content to the same account within 24 h with
   HTTP 409, and a retry would post twice anyway):
   - Instagram: `node .claude/skills/publish-reel/publish.mjs reel renders/<name>-ig.mp4 renders/<name>-ig.txt --cover-ms=N --audio-name="Midas - <Character>"`
   - YouTube: the step 5 command without `--dry-run`. It prints the post id and the URL; without
     a URL yet: `zernio.mjs wait <post id>` (background).
   - TikTok: the step 5 command without `--dry-run`. The draft is forced (`tiktokSettings.draft`
     at the root and on the entry; no flag turns it off) and the script errors out if TikTok's
     answer lacks the draft markers. `published` here means handed to the inbox.
8. **Log it**: a row in `renders/PUBLISHED.md` (date, model, film, both links). StlViewer keeps
   its own log of the same account in its `render-guess/results.md`.
9. **Report** the Instagram permalink, the YouTube URL, and for TikTok: "draft waiting in the
   TikTok app (inbox notification): add sound, paste the caption and hashtags, post" + two code
   blocks: the caption without hashtags, then the hashtags (max 5) on their own.

## X (the user's own profile)

On X the user posts under their own name, **@d_o_m_e_r**, not as @kingmidasmodels, through vidIQ's
MCP (`vidiq_social_account_list` platform X, then a draft, then publish; X was connected for
it on 2026-11-17). The film is the `-ig` one (with music). The text, `renders/<name>-x.txt`,
**has no fluff** (the user's rule, 2026-11-17): no hook, no call to action, no emoji. It gives the
render's technical facts, read from its log: model and triangle count, GPU, engine (OptiX on RT
cores, own Kotlin renderer), traced resolution and spp, bounces, denoiser and upscale (albedo +
normal guides), output resolution, time per frame and frames, then the model credit with its
link. The limit is 280 characters by X's weighting: a URL counts 23, while `→` and emoji count 2,
so write `->`. The post goes into the same approval as the other platforms.

The film goes up with `vidiq_video_upload` `localFile`, using a POSIX path (`/c/Users/...`); a
Windows path is refused. Run the command it returns in Bash, then call it again with the
`uploadId`. Then `vidiq_social_post_draft_create` (media = that uploadId, destination X with the
account's `platformAccountId`), approval, and `vidiq_social_post_publish` with the postId and
revision. **Cost:** vidIQ charges 5 credits for an X post and 40 for one with a link in the text,
on a plan of 150 credits a month (115 left on 2026-11-17). Tell the user the cost, and offer the
credit without the URL ("on Printables"), which costs 5.

## Limits and errors

- TikTok: **5 pending drafts per account**; a 6th fails, and waiting does not free a slot - the
  user posts or deletes drafts in the app. TikTok's "too many posts in the last 24 hours via the
  API" counts drafts too.
- TikTok drafts arrive in the app **with a long delay**: tell the user, and never send one again
  because it hasn't shown up yet (each send takes a slot).
- Zernio answers HTTP 207 when a platform fails: `zernio.mjs` reads `post.status` and exits 1 with
  the `errorMessage`.
- YouTube: Pixabay music can draw a Content ID claim (the video stays up); the credit line stays
  anyway. Title, tags and thumbnail can be changed later through Zernio's "Update post metadata",
  the description through "Edit post".
- Instagram specifics (token, music, cover): `../publish-reel/SKILL.md`; TikTok alone:
  `../publish-tiktok/SKILL.md`.
