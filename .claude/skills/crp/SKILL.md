---
name: crp
description: Commit, release, push - ship the current work of KoltinRayTracer in one go - commit what is pending, run the tests, bump the version (pom.xml + CHANGELOG.md), tag vX.Y.Z, push the branch and the tag, merge into the default branch (main - the user's "master") and push it, then publish the release. Use when the user says "crp", "/crp", "zrób release", "wydaj wersję", "commit release push", "wypuść", "zmerguj na mastera/maina", or asks to tag and publish the work.
---

# CRP: commit, release, push

One run takes the work on the current branch all the way to the default branch with a version
tag. Invoking it is the user's go-ahead for every step below, pushes and the merge included;
stop and report at the first step that fails instead of forcing it.

**The target branch is `main`**, the repository's default (`git symbolic-ref
refs/remotes/origin/HEAD`). When the user says "master" they mean that one: `origin/master` is a
stale branch from 2025-11 ("finished ray tracing in one weekend") - never merge into it unless
the user names `origin/master` explicitly and confirms.

## 1. Look before shipping

```
git fetch origin
git status --short
git log --oneline origin/main..HEAD
git describe --tags --abbrev=0        # the last release, if any
```

- The working tree always holds untracked renders and scratch files that are **never**
  committed: `viewer-*.png|ppm`, `orc-*`, `image.ppm`, `cp.txt`, `bench-out/`, `models/`,
  `renders/` (films, captions, music), `reference/**/*.mp4`. Stage files by name, never
  `git add -A` / `git add .` at the root.
- If `HEAD` is `main` itself and there is nothing new on it since the last tag, there is
  nothing to release - say so and stop.
- If the branch has fallen behind `origin/main` (`git log HEAD..origin/main` not empty), merge
  `origin/main` into it first and run the tests on the result.

## 2. Commit what is pending

Commit the work that belongs to the release, in commits that each say one thing (AGENTS.md:
short imperative titles under ~72 characters, a body that says why; no refactor bundled with a
feature). End every message with the attribution lines the current system reminder gives.
Never `--no-verify`, never amend a pushed commit, never commit a token or secret.

## 3. Test

```
mvn -o test            # drop -o when the pom's dependencies changed
mvn -o -q clean package
```

All tests must pass. `GpuCpuComparisonTest` and `ImageRegressionTest` need the CUDA card; if
they are skipped (no GPU), say so in the report - a release without them is untested on the
GPU. A failure stops the run: fix it in its own commit, or report it and leave the release.

## 4. Version and changelog

Semantic versions, tags `vX.Y.Z`. The version lives in `pom.xml` (`<version>`, line ~7).
Pick the bump from what changed since the last tag (`git log <last-tag>..HEAD --oneline`):

| Change | Bump |
|---|---|
| Something users relied on removed or changed incompatibly (a backend, a `-D` switch, a file format such as `.rtscene`) | major |
| A new feature, scene, switch or skill | minor |
| Only fixes, speed-ups, docs, tests | patch |

With no tag yet, the first release takes the pom's version as it stands if that version was
never released, otherwise the bump from it. If the bump is not obvious, take the larger one and
say why in the report.

Then:
1. Set `<version>` in `pom.xml` to the new version.
2. Add a section at the top of `CHANGELOG.md` (create the file with a `# Changelog` heading if
   it does not exist):
   ```
   ## vX.Y.Z - YYYY-MM-DD

   - one line per user-visible change, newest work first, in the repo's plain style
     (what changed and why it matters; numbers where measured)
   ```
   Write it from the commits, not by pasting their titles; leave out pure housekeeping.
3. Commit both: `release vX.Y.Z` (with the attribution lines).
4. Tag it, annotated, with the changelog section as the message:
   `git tag -a vX.Y.Z -F <file with the section>` (or `-m`).

## 5. Push and merge

```
git push origin <branch>
git push origin vX.Y.Z
git switch main
git pull --ff-only origin main
git merge --ff-only <branch>  ||  git merge --no-ff <branch> -m "merge <branch> for vX.Y.Z"
mvn -o test                    # only when the merge was not a fast-forward
git push origin main
git switch <branch>            # back where the work happens
```

- A fast-forward is preferred: then `main` points at the tagged release commit.
- A merge conflict, or tests failing on the merged result, stops the run: abort the merge
  (`git merge --abort`), push nothing to `main`, report what conflicted.
- Never force-push, never rewrite `main`, never delete a branch or a tag.
- This worktree may not be the only one: if `git switch main` refuses because `main` is checked
  out in another worktree, update it from here instead -
  `git push origin <branch>:main` after checking `git merge-base --is-ancestor origin/main <branch>`
  (a fast-forward on the remote), or report it.

## 6. Publish the release

- With the GitHub CLI (`gh --version` works and `gh auth status` is logged in):
  `gh release create vX.Y.Z --title "vX.Y.Z" --notes-file <the changelog section>`.
- Without it (the case on this machine as of 2026-11-02) the pushed annotated tag is the
  release; give the user the link to turn it into a GitHub Release by hand:
  `https://github.com/domer-sz/KoltinRayTracer/releases/new?tag=vX.Y.Z`.
  (GitHub says the repository moved to `domer-sz/KotlinRayTracer`; the old name redirects.)

## 7. Report

One short block: the version and why that bump, the commits that went in
(`git log --oneline <previous tag>..vX.Y.Z`), test results (and whether the GPU tests ran),
that the branch, the tag and `main` are pushed (fast-forward or merge commit), and the release
link. If any step stopped the run, say which, what state is left (e.g. tag pushed, `main` not
merged) and what is needed to finish.
