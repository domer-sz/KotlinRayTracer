// Puts a music track under a rendered film: the picture is copied as it is, the track is cut
// to the film's length (from START seconds in), faded in and out and brought to Instagram's
// loudness (-14 LUFS). Instagram then treats it as the Reel's original audio.
// Taken over unchanged from StlViewer's publish-reel skill.
//
//   node add-music.mjs <film.mp4> <music.(mp3|wav|m4a|...)> [out.mp4] [--start=S|auto] [--fade-out=S]
//
// --start=auto picks the loudest stretch of the track as long as the film (skips quiet intros).
//
// Default out: <film>-music.mp4 next to the film. Needs ffmpeg/ffprobe on the PATH (scoop).
import { execFileSync, spawnSync } from 'node:child_process';

const args = process.argv.slice(2);
const flags = Object.fromEntries(args.filter((a) => a.startsWith('--')).map((a) => a.slice(2).split('=')));
const [film, music, outArg] = args.filter((a) => !a.startsWith('--'));
if (!film || !music) {
  console.log('usage: add-music.mjs <film.mp4> <music> [out.mp4] [--start=S] [--fade-out=S]');
  process.exit(1);
}
const out = outArg || film.replace(/\.mp4$/i, '') + '-music.mp4';
const fadeOut = parseFloat(flags['fade-out'] || '1.5');

const probe = (file, entry) => execFileSync('ffprobe', ['-v', 'error', '-show_entries', entry, '-of', 'default=nw=1:nk=1', file]).toString().trim();
const duration = parseFloat(probe(film, 'format=duration'));
const trackLength = parseFloat(probe(music, 'format=duration'));

/** The second where the loudest stretch as long as the film begins (momentary loudness, ebur128). */
function loudestStart() {
  // ebur128 logs to stderr.
  const log = spawnSync('ffmpeg', ['-nostats', '-i', music, '-af', 'ebur128', '-f', 'null', '-'], { maxBuffer: 1 << 28 }).stderr.toString();
  const sum = [], count = [];
  for (const m of log.matchAll(/t: *([\d.]+)[^\n]*?M: *(-?[\d.]+)/g)) {
    const s = Math.floor(+m[1]), v = +m[2];
    if (v > -70) { sum[s] = (sum[s] || 0) + v; count[s] = (count[s] || 0) + 1; }
  }
  const L = sum.map((v, s) => v / count[s]), span = Math.ceil(duration);
  let best = -Infinity, at = 0;
  for (let s = 0; s + span <= L.length; s++) {
    const w = L.slice(s, s + span).filter((v) => v !== undefined);
    const mean = w.reduce((a, b) => a + b, 0) / w.length;
    if (w.length >= span - 1 && mean > best) { best = mean; at = s; }
  }
  return at;
}
const start = flags.start === 'auto' ? loudestStart() : parseFloat(flags.start || '0');
if (trackLength - start < duration) console.log(`note: the track has ${(trackLength - start).toFixed(1)} s from ${start} s, the film ${duration.toFixed(1)} s - the end will be silent`);

const audio = [
  `atrim=start=${start}:duration=${duration}`, 'asetpts=PTS-STARTPTS',
  'afade=t=in:st=0:d=0.4', `afade=t=out:st=${Math.max(0, duration - fadeOut)}:d=${fadeOut}`,
  'loudnorm=I=-14:TP=-1.5:LRA=11', 'aresample=48000',
].join(',');
execFileSync('ffmpeg', ['-v', 'error', '-y', '-i', film, '-i', music,
  '-filter_complex', `[1:a]${audio}[a]`, '-map', '0:v', '-map', '[a]',
  '-c:v', 'copy', '-c:a', 'aac', '-b:a', '192k', '-ac', '2', '-t', String(duration), '-movflags', '+faststart', out], { stdio: 'inherit' });
console.log(`${out}: ${duration.toFixed(1)} s, music from ${start} s of ${music}`);
