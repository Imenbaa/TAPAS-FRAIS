from textgrid import TextGrid, IntervalTier

def get_textgrid_transcription(tg_path, tier_name):
    tg = TextGrid.fromFile(tg_path)

    # Afficher les tiers disponibles (debug)
    available_tiers = [t.name for t in tg.tiers]

    # Chercher le bon tier
    tier = None
    for t in tg.tiers:
        if t.name.strip().lower() == tier_name.strip().lower():
            tier = t
            break

    if tier is None:
        raise ValueError(
            f"Tier '{tier_name}' introuvable. "
            f"Tiers disponibles: {available_tiers}"
        )

    if not isinstance(tier, IntervalTier):
        raise ValueError(f"Le tier '{tier.name}' n'est pas un IntervalTier")

    words = [
        interval.mark.strip()
        for interval in tier.intervals
        if interval.mark.strip()
    ]

    return " ".join(words)
