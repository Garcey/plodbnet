# Raw rank crops (not loaded)

Whole-corner crops of each board rank (`<rank>_board.png`, `<rank>_board1.png`)
kept as reference material for re-harvesting glyphs. The classifier loads only
`../rank_<R>_*.png` (`cards._load_templates`); new templates come from
labeled frames via `python -m plo5bp.ocr.tools.label_cards` (hero-hole crops
are de-rotated to their slot's calibrated angle first, like the runtime).
