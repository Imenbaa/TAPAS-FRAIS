import os

def list_files(trans_dir):
    tg_to_wav = {}
    for f in sorted(os.listdir(trans_dir)):
        if f.endswith(".TextGrid") and not f.endswith("pr_analyse.TextGrid"):
            if "Rhapsodie" in trans_dir:
                tg_to_wav[f] = f.split("-")[0] + "-" + f.split("-")[1] + ".wav"
            else:
                tg_to_wav[f] = f.split(".")[0] + ".wav"
    return tg_to_wav