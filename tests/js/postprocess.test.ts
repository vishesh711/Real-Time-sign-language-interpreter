/**
 * Property-based and unit tests for spell-check utilities.
 *
 * Feature: sign-language-interpreter, Property 10: Spell-check idempotence on dictionary words
 * Validates: Requirements 5.8
 */

import * as fc from 'fast-check';
import { describe, it, expect } from 'vitest';
import { spellCheck } from '../../browser/postprocess';

// ---------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------

/**
 * Generate a non-empty uppercase ASCII word (A–Z only, 1–10 chars).
 * Mirrors the kind of words produced by the ASL fingerspelling pipeline.
 */
const uppercaseWord = (): fc.Arbitrary<string> =>
  fc
    .array(
      fc.integer({ min: 65, max: 90 }).map((c) => String.fromCharCode(c)),
      { minLength: 1, maxLength: 10 },
    )
    .map((chars) => chars.join(''));

/**
 * Generate a Set of 1–8 non-empty uppercase words for use as a dictionary.
 */
const uppercaseDictionary = (): fc.Arbitrary<Set<string>> =>
  fc
    .array(uppercaseWord(), { minLength: 1, maxLength: 8 })
    .map((words) => new Set(words));

// ---------------------------------------------------------------------------
// Property 10a — dictionary word is returned unchanged
// ---------------------------------------------------------------------------

describe('spellCheck — Property 10a: dictionary-word idempotence', () => {
  it(
    /** Feature: sign-language-interpreter, Property 10: Spell-check idempotence on dictionary words */
    'Property 10a: any word already in the dictionary is returned unchanged',
    () => {
      fc.assert(
        fc.property(
          uppercaseWord(),
          uppercaseDictionary(),
          (word, dict) => {
            // Make sure the word is in the dictionary
            const dictWithWord = new Set([...dict, word]);

            const result = spellCheck(word, dictWithWord);

            // Must come back unchanged since it's already in the dictionary
            expect(result).toBe(word);
          },
        ),
        { numRuns: 100 },
      );
    },
  );
});

// ---------------------------------------------------------------------------
// Property 10b — applying spell-check twice equals applying it once
// ---------------------------------------------------------------------------

describe('spellCheck — Property 10b: result idempotence', () => {
  it(
    /** Feature: sign-language-interpreter, Property 10: Spell-check idempotence on dictionary words */
    'Property 10b: spellCheck(spellCheck(w, dict), dict) === spellCheck(w, dict)',
    () => {
      fc.assert(
        fc.property(
          uppercaseWord(),
          uppercaseDictionary(),
          (word, dict) => {
            const once = spellCheck(word, dict);
            const twice = spellCheck(once, dict);

            // Applying spell-check a second time must not change the result
            expect(twice).toBe(once);
          },
        ),
        { numRuns: 100 },
      );
    },
  );
});

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

describe('spellCheck — unit tests', () => {
  it(
    /** Feature: sign-language-interpreter, Property 10: Spell-check idempotence on dictionary words */
    'corrects "NANE" to "NAME" (edit distance 1)',
    () => {
      const dict = new Set(['NAME', 'NONE']);
      expect(spellCheck('NANE', dict)).toBe('NAME');
    },
  );

  it(
    /** Feature: sign-language-interpreter, Property 10: Spell-check idempotence on dictionary words */
    'corrects "HELO" to "HELLO" (edit distance 1)',
    () => {
      const dict = new Set(['HELLO', 'HELP']);
      expect(spellCheck('HELO', dict)).toBe('HELLO');
    },
  );

  it(
    /** Feature: sign-language-interpreter, Property 10: Spell-check idempotence on dictionary words */
    'returns "HELLO" unchanged when it is already in the dictionary',
    () => {
      const dict = new Set(['HELLO', 'HELP']);
      expect(spellCheck('HELLO', dict)).toBe('HELLO');
    },
  );
});
