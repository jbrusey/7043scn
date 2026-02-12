
# Example:
def levenshtein_dp(x: str, y: str) -> int:
    m, n = len(x), len(y)

    # Create DP table
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill table row by row
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if x[i - 1] == y[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,      # deletion
                dp[i][j - 1] + 1,      # insertion
                dp[i - 1][j - 1] + cost  # substitution
            )

    return dp


# Example
print(levenshtein_dp("kitten", "sitting"))  
