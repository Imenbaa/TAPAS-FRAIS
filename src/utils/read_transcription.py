from textgrid import TextGrid

def get_textgrid_transcription(tg_path, tier_name="transcription"):
    """
    Read a TextGrid file and return the concatenated transcription from the specified tier.
    """
    tg = TextGrid.fromFile(tg_path)

    # Find the tier
    tier = None
    for t in tg.tiers:
        if t.name.lower() == tier_name.lower():
            tier = t
            break

    if tier is None:
        # fallback: take first tier
        tier = tg.tiers[0]

    # Concatenate all non-empty intervals
    words = [
        interval.mark.strip()
        for interval in tier.intervals
        if interval.mark and interval.mark.strip()
    ]

    transcription = " ".join(words)
    return transcription
