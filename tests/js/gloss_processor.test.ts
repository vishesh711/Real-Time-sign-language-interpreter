/**
 * Property-based and unit tests for GlossPostProcessor.
 *
 * Feature: sign-language-interpreter, Property 5: Gloss-to-English returns non-empty string
 * Validates: Requirements 3.4, 5.9
 *
 * Property 5 — Gloss-to-English returns non-empty string:
 *   For any non-empty sequence of valid ASL gloss tokens, the GlossPostProcessor
 *   SHALL return a non-empty string.
 */

import * as fc from 'fast-check';
import { describe, it, expect } from 'vitest';
import { GlossPostProcessor } from '../../browser/gloss_processor';

// ---------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------

/**
 * Generates a single uppercase ASCII word (1–10 chars, A–Z only).
 * Represents a valid ASL gloss token.
 */
const glossTokenArb = fc.stringOf(
  fc.mapToConstant(
    { num: 26, build: (i) => String.fromCharCode(65 + i) }, // A–Z
  ),
  { minLength: 1, maxLength: 10 },
);

/**
 * Generates a non-empty array of 1–10 valid gloss tokens.
 */
const nonEmptyGlossArrayArb = fc.array(glossTokenArb, {
  minLength: 1,
  maxLength: 10,
});

// ---------------------------------------------------------------------------
// Property 5a: non-empty input → non-empty output (model not loaded)
// ---------------------------------------------------------------------------

describe('GlossPostProcessor', () => {
  it(
    /** Feature: sign-language-interpreter, Property 5: Gloss-to-English returns non-empty string */
    'Property 5a: non-empty gloss input returns non-empty string when model is not loaded',
    async () => {
      await fc.assert(
        fc.asyncProperty(nonEmptyGlossArrayArb, async (glosses) => {
          // Create a fresh unloaded processor (isLoaded = false by default)
          const processor = new GlossPostProcessor();
          const result = await processor.glossToEnglish(glosses);

          // Must be a non-empty string
          expect(typeof result).toBe('string');
          expect(result.length).toBeGreaterThan(0);
          // Fallback behaviour: space-joined glosses
          expect(result).toBe(glosses.join(' '));
        }),
        { numRuns: 100 },
      );
    },
  );

  // -------------------------------------------------------------------------
  // Property 5b: empty input → empty output
  // -------------------------------------------------------------------------

  it(
    /** Feature: sign-language-interpreter, Property 5: Gloss-to-English returns non-empty string */
    'Property 5b: empty gloss input returns empty string',
    async () => {
      const processor = new GlossPostProcessor();
      const result = await processor.glossToEnglish([]);
      expect(result).toBe('');
    },
  );

  // -------------------------------------------------------------------------
  // Unit test: loaded pipeline returns translation
  // -------------------------------------------------------------------------

  it(
    /** Feature: sign-language-interpreter, Property 5: Gloss-to-English returns non-empty string */
    'Unit: loaded pipeline — glossToEnglish returns the translation_text from the model',
    async () => {
      const processor = new GlossPostProcessor();

      // Mock the private pipeline via `as any` to simulate a loaded model
      const mockPipeline = async (_input: string) => ({
        translation_text: 'I want to eat',
      });
      (processor as any).pipeline = mockPipeline;
      processor.isLoaded = true;

      const result = await processor.glossToEnglish(['WANT', 'EAT']);
      expect(result).toBe('I want to eat');
    },
  );

  // -------------------------------------------------------------------------
  // Unit test: loaded pipeline returns empty translation → fallback to raw glosses
  // -------------------------------------------------------------------------

  it(
    /** Feature: sign-language-interpreter, Property 5: Gloss-to-English returns non-empty string */
    'Unit: loaded pipeline returning empty translation falls back to space-joined glosses',
    async () => {
      const processor = new GlossPostProcessor();

      // Mock pipeline that returns an empty translation_text
      const mockPipeline = async (_input: string) => ({
        translation_text: '',
      });
      (processor as any).pipeline = mockPipeline;
      processor.isLoaded = true;

      const result = await processor.glossToEnglish(['WANT', 'EAT']);
      expect(result).toBe('WANT EAT');
    },
  );
});
