# Design pattern: elements (Elements & animation)

Status: ready. The default and currently the only implemented pattern for Engine B (Verticals).

- Format: vertical 9:16 Short, 720x1280 to 1080x1920, 45-75 s, never upscaled.
- Visuals: one shot every few seconds from b-roll stills (Wikimedia PD/CC0, Pixabay), AI stills
  (Cloudflare FLUX, NVIDIA FLUX fallback) and gradient cards, chosen per script beat.
- Motion: every still moves (Ken Burns pan/zoom, living backgrounds, motion blocks); no frame stays
  static longer than 3.5 s.
- Elements: animated text blocks, numbers, labels and simple shapes from the style guide
  (`style/`), layered over the b-roll; no recurring people or mascots.
- Captions: burned-in word-timed captions (Whisper), Montserrat bold with DejaVu Sans Bold fallback.
- Voice: Kokoro-82M (default am_michael), paced 150-165 WPM; Edge TTS fallback.
- Music: quiet CC0 Open Lo-Fi bed under the voice (about -26 dB).
- Gates: no-personal-names check, visual judge and QC; AI visuals set containsSyntheticMedia.
- Pattern `characters` (recurring characters) is coming soon; until then any request for it,
  or for an unknown pattern, is logged and rendered as `elements`.
