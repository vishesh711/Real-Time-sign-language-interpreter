/**
 * Fingerspelling post-processing utilities.
 *
 * Provides two exported functions:
 *   - `collapseAndSegment`: collapses repeated-letter runs and segments a
 *     per-frame letter stream into words using silence gaps.
 *   - `spellCheck`: applies Levenshtein edit-distance correction against a
 *     vocabulary dictionary, handling common ASL classifier confusions
 *     (A↔E, M↔N, S↔A).
 *
 * Requirements: 5.7, 5.8
 */

// ---------------------------------------------------------------------------
// collapseAndSegment
// ---------------------------------------------------------------------------

/**
 * Collapse a stream of per-frame letter predictions into segmented words.
 *
 * Algorithm:
 *   1. Walk through `letterStream` frame by frame.
 *   2. Track the current run: a (letter, length) pair for consecutive
 *      identical non-null frames.
 *   3. When the letter changes OR a null frame is seen, flush the current run:
 *      if its length ≥ `minLetterHold`, append the letter to the active word
 *      accumulator.
 *   4. Track consecutive null frames. When their count reaches `pauseThreshold`,
 *      flush the active word accumulator as a completed word (if non-empty).
 *
 * Requirements: 5.7
 *
 * @param letterStream   Per-frame predictions — each element is a letter
 *                       string or `null` (no hand / silent frame).
 * @param pauseThreshold Number of consecutive `null` frames that constitute a
 *                       word boundary (default 8).
 * @param minLetterHold  Minimum consecutive identical-letter frames required
 *                       for the letter to be included in the output (default 4).
 * @returns              Array of reconstructed word strings, each being the
 *                       concatenation of collapsed letters for that segment.
 */
export function collapseAndSegment(
  letterStream: (string | null)[],
  pauseThreshold = 8,
  minLetterHold = 4,
): string[] {
  const words: string[] = [];

  // Letters accumulated for the current word-in-progress.
  let currentWordLetters: string[] = [];

  // Current run tracking.
  let runLetter: string | null = null;
  let runLength = 0;

  // Consecutive null frame counter.
  let nullCount = 0;

  /**
   * Flush the active letter run. If the run length meets `minLetterHold`,
   * append the run letter to the current word.
   */
  function flushRun(): void {
    if (runLetter !== null && runLength >= minLetterHold) {
      currentWordLetters.push(runLetter);
    }
    runLetter = null;
    runLength = 0;
  }

  /**
   * Flush the current word accumulator as a completed word (if non-empty).
   */
  function flushWord(): void {
    if (currentWordLetters.length > 0) {
      words.push(currentWordLetters.join(''));
      currentWordLetters = [];
    }
  }

  for (const frame of letterStream) {
    if (frame === null) {
      // Null frame: end any active letter run first.
      flushRun();
      nullCount += 1;

      // Once we hit the pause threshold, segment the word.
      if (nullCount === pauseThreshold) {
        flushWord();
      }
    } else {
      // Non-null frame: reset the null counter.
      nullCount = 0;

      if (frame === runLetter) {
        // Continue the current run.
        runLength += 1;
      } else {
        // Letter changed — flush old run and start a new one.
        flushRun();
        runLetter = frame;
        runLength = 1;
      }
    }
  }

  // End-of-stream: flush any remaining run and word.
  flushRun();
  flushWord();

  return words;
}

// ---------------------------------------------------------------------------
// spellCheck
// ---------------------------------------------------------------------------

/**
 * Compute the Levenshtein edit distance between two strings.
 *
 * Uses the standard dynamic-programming approach with O(min(a,b)) space.
 *
 * @param a First string.
 * @param b Second string.
 * @returns  The minimum number of single-character insertions, deletions, or
 *           substitutions required to transform `a` into `b`.
 */
function levenshtein(a: string, b: string): number {
  // Ensure `a` is the shorter string to minimise memory usage.
  if (a.length > b.length) {
    [a, b] = [b, a];
  }

  const m = a.length;
  const n = b.length;

  // `prev[j]` = edit distance between a[0..i-1] and b[0..j-1].
  let prev = Array.from({ length: m + 1 }, (_, j) => j);

  for (let i = 1; i <= n; i++) {
    const curr: number[] = new Array(m + 1);
    curr[0] = i;
    for (let j = 1; j <= m; j++) {
      if (b[i - 1] === a[j - 1]) {
        curr[j] = prev[j - 1];
      } else {
        curr[j] = 1 + Math.min(prev[j - 1], prev[j], curr[j - 1]);
      }
    }
    prev = curr;
  }

  return prev[m];
}

/**
 * Apply edit-distance spell correction to a fingerspelled word.
 *
 * - If `word` is already in `dictionary`, return it unchanged (idempotent).
 * - Otherwise find the dictionary entry with the smallest Levenshtein distance.
 * - Return the closest match only if its edit distance is ≤ 2; otherwise
 *   return the original word unmodified.
 *
 * This corrects common ASL classifier confusions such as A↔E, M↔N, and S↔A,
 * which typically produce edit distances of 1 or 2.
 *
 * Requirements: 5.8
 *
 * @param word        The fingerspelled word to correct (case-sensitive).
 * @param dictionary  The vocabulary to search for the closest match.
 * @returns           The corrected word if a close match exists, else `word`.
 */
export function spellCheck(word: string, dictionary: Set<string>): string {
  // Fast path: already in dictionary.
  if (dictionary.has(word)) {
    return word;
  }

  let bestWord = word;
  let bestDist = Infinity;

  for (const candidate of dictionary) {
    const dist = levenshtein(word, candidate);
    if (dist < bestDist) {
      bestDist = dist;
      bestWord = candidate;
    }
  }

  // Only accept corrections within edit distance 2.
  return bestDist <= 2 ? bestWord : word;
}
