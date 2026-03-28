def mergeTranscripts(transcripts, maxWindow=20):

    if not transcripts:
        return ""

    mergedWords = transcripts[0].split()

    for i in range(1, len(transcripts)):

        currWords = transcripts[i].split()

        overlapSize = 0

        # try matching last k words of previous with first k of current
        maxCheck = min(len(mergedWords), len(currWords), maxWindow)

        for k in range(maxCheck, 0, -1):
            if mergedWords[-k:] == currWords[:k]:
                overlapSize = k
                break

        # add only non-overlapping words
        mergedWords.extend(currWords[overlapSize:])

    return " ".join(mergedWords)