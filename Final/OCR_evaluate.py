import re

def levenshtein(a, b):
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


def normalize(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def calculate_metrics(gt, hyp):
    # CER
    gt_n, hyp_n = normalize(gt), normalize(hyp)
    dist_c = levenshtein(gt_n, hyp_n)
    cer_score = dist_c / max(len(gt_n), 1)

    # WER
    gt_w, hyp_w = gt_n.split(), hyp_n.split()
    dist_w = levenshtein(gt_w, hyp_w)
    wer_score = dist_w / max(len(gt_w), 1)

    return cer_score, wer_score


# รันภายใน
if __name__ == '__main__':
    calculate_metrics('', '')