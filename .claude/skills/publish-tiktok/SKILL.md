---
name: publish-tiktok
description: Send a rendered film from this repo's renders/ to the @kingmidasmodels TikTok app as a DRAFT (to the TikTok inbox, through the Zernio posting API), then give the English caption and, in a separate block, max 5 proposed hashtags (TikTok drops captions on drafts). For Instagram + YouTube + TikTok together use publish-film. Use when the user asks to send / wyślij / wrzuć a render, a film or "model N" to TikTok (as a draft, "szkic"). On this PC use it instead of the synced tiktok-draft-publisher skill, which needs Python.
---

# Sending a film to TikTok as a draft

Taken over from StlViewer's `publish-tiktok` skill (2026-11-17), its working route only: the
draft goes through **Zernio** (zernio.com), a posting API with its own TikTok-approved app, with
`../publish-film/zernio.mjs` (shared with YouTube; `../publish-film/SKILL.md` has the full flow
and the limits). StlViewer's own sandbox TikTok app (`tiktok.mjs`) never delivered a draft to the
public account and was not copied.

1. **Pick the film.** The silent `renders/<name>-tiktok.mp4` (made from the render as in
   `../publish-reel/SKILL.md` step 3 when it is missing); the sound is added in the TikTok app.
2. **Caption** `renders/<name>-tiktok.txt`, as in `../publish-film/SKILL.md` step 4 (the
   attribution from the model's `INFO.md`, no music line, no line on how it was made, max 5
   hashtags as the last line).
3. `node .claude/skills/publish-film/zernio.mjs tiktok-draft renders/<name>-tiktok.mp4 --caption=renders/<name>-tiktok.txt --dry-run`,
   then the same without `--dry-run` - **once**: every send becomes its own draft and takes one
   of the 5 slots. The draft is forced; the script fails if TikTok's answer lacks the draft
   markers. `published` = TikTok accepted it into the inbox (`v_inbox...` id, no post URL).
4. **Tell the user** the draft is on its way to the TikTok app (it can take a while to show up)
   and give **two separate code blocks** to paste there (the user's rule since 2026-11-03):
   block 1 = the caption without its hashtag line, block 2 = the hashtags alone (at most 5).

A draft is not public, so it needs no approval round like a Reel's - but it takes a slot, so
send it only when the user asked for TikTok (or in publish-film's approved go).

- Zernio key: StlViewer's `~/.config/stlviewer/zernio.json`; never print, copy or commit it.
- 5 pending drafts per account; a 6th fails until the user posts or deletes one in the app.
