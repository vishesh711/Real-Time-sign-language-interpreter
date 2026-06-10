/**
 * Property-based and unit tests for OutputBuffer.
 *
 * Feature: sign-language-interpreter, Property 7: Transcript append invariant
 * Feature: sign-language-interpreter, Property 11: Backspace decrements committed length
 * Validates: Requirements 5.3, 5.4, 5.10
 *
 * Property 7 — Transcript append invariant:
 *   For any OutputBuffer state with N committed entries, performing one
 *   accept(word) operation SHALL result in exactly N+1 committed entries,
 *   with the new entry at position N equal to the accepted word.
 *
 * Property 11 — Backspace decrements committed length:
 *   For any OutputBuffer with N > 0 committed entries, calling backspace()
 *   SHALL result in exactly N−1 committed entries. Calling backspace() on an
 *   empty buffer SHALL leave the buffer unchanged (length remains 0).
 */

import * as fc from 'fast-check';
import { describe, it, expect, beforeEach } from 'vitest';
import { OutputBuffer } from '../../browser/output_buffer';

// ---------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------

/** A non-empty word string (printable ASCII, 1–20 chars). */
const wordArb = fc.string({ minLength: 1, maxLength: 20 });

/** An array of 0–20 words representing pre-committed buffer state. */
const wordListArb = fc.array(wordArb, { minLength: 0, maxLength: 20 });

/** A non-empty word list (at least 1 committed entry). */
const nonEmptyWordListArb = fc.array(wordArb, { minLength: 1, maxLength: 20 });

/**
 * Build an OutputBuffer pre-populated with the given words via accept().
 */
function bufferWith(words: string[]): OutputBuffer {
  const buf = new OutputBuffer();
  for (const w of words) {
    buf.accept(w);
  }
  return buf;
}

// ---------------------------------------------------------------------------
// Property 7 — Transcript append invariant
// ---------------------------------------------------------------------------

describe('OutputBuffer — Property 7: Transcript append invariant', () => {
  it(
    /** Feature: sign-language-interpreter, Property 7: Transcript append invariant */
    'Property 7: accept(word) adds exactly one entry at the end of committed',
    () => {
      fc.assert(
        fc.property(wordListArb, wordArb, (existing, word) => {
          const buf = bufferWith(existing);
          const nBefore = buf.committed.length;

          buf.accept(word);

          expect(buf.committed.length).toBe(nBefore + 1);
          expect(buf.committed[nBefore]).toBe(word);
        }),
        { numRuns: 200 },
      );
    },
  );

  it(
    /** Feature: sign-language-interpreter, Property 7: Transcript append invariant */
    'Property 7: accept(word) clears the candidate',
    () => {
      fc.assert(
        fc.property(wordListArb, wordArb, fc.option(wordArb, { nil: null }), (existing, word, candidate) => {
          const buf = bufferWith(existing);
          buf.setCandidate(candidate);

          buf.accept(word);

          expect(buf.candidate).toBeNull();
        }),
        { numRuns: 200 },
      );
    },
  );

  it(
    /** Feature: sign-language-interpreter, Property 7: Transcript append invariant */
    'Property 7: accept(word) also appends to glossBuffer',
    () => {
      fc.assert(
        fc.property(wordListArb, wordArb, (existing, word) => {
          const buf = bufferWith(existing);
          const glossBefore = buf.glossBuffer.length;

          buf.accept(word);

          expect(buf.glossBuffer.length).toBe(glossBefore + 1);
          expect(buf.glossBuffer[glossBefore]).toBe(word);
        }),
        { numRuns: 200 },
      );
    },
  );
});

// ---------------------------------------------------------------------------
// Property 11 — Backspace decrements committed length
// ---------------------------------------------------------------------------

describe('OutputBuffer — Property 11: Backspace decrements committed length', () => {
  it(
    /** Feature: sign-language-interpreter, Property 11: Backspace decrements committed length */
    'Property 11: backspace() on non-empty buffer removes exactly the last entry',
    () => {
      fc.assert(
        fc.property(nonEmptyWordListArb, (words) => {
          const buf = bufferWith(words);
          const nBefore = buf.committed.length;

          buf.backspace();

          expect(buf.committed.length).toBe(nBefore - 1);
          // Remaining entries are unchanged
          for (let i = 0; i < buf.committed.length; i++) {
            expect(buf.committed[i]).toBe(words[i]);
          }
        }),
        { numRuns: 200 },
      );
    },
  );

  it(
    /** Feature: sign-language-interpreter, Property 11: Backspace decrements committed length */
    'Property 11: backspace() on empty buffer is a no-op (length stays 0)',
    () => {
      const buf = new OutputBuffer();
      buf.backspace();
      expect(buf.committed.length).toBe(0);
    },
  );

  it(
    /** Feature: sign-language-interpreter, Property 11: Backspace decrements committed length */
    'Property 11: repeated backspace() eventually empties the buffer',
    () => {
      fc.assert(
        fc.property(wordListArb, (words) => {
          const buf = bufferWith(words);

          for (let i = 0; i < words.length; i++) {
            buf.backspace();
          }

          expect(buf.committed.length).toBe(0);
        }),
        { numRuns: 200 },
      );
    },
  );
});

// ---------------------------------------------------------------------------
// Unit tests — clear() resets all state (Requirement 5.4)
// ---------------------------------------------------------------------------

describe('OutputBuffer — clear()', () => {
  it('clear() resets committed, candidate, and glossBuffer', () => {
    const buf = new OutputBuffer();
    buf.accept('HELLO');
    buf.accept('WORLD');
    buf.setCandidate('TEST');

    buf.clear();

    expect(buf.committed).toEqual([]);
    expect(buf.candidate).toBeNull();
    expect(buf.glossBuffer).toEqual([]);
  });

  it('clear() on an empty buffer is a no-op', () => {
    const buf = new OutputBuffer();
    buf.clear();
    expect(buf.committed).toEqual([]);
    expect(buf.candidate).toBeNull();
    expect(buf.glossBuffer).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// Unit tests — setCandidate() (Requirement 5.6)
// ---------------------------------------------------------------------------

describe('OutputBuffer — setCandidate()', () => {
  it('setCandidate(word) stores the candidate', () => {
    const buf = new OutputBuffer();
    buf.setCandidate('PENDING');
    expect(buf.candidate).toBe('PENDING');
  });

  it('setCandidate(null) clears the candidate', () => {
    const buf = new OutputBuffer();
    buf.setCandidate('PENDING');
    buf.setCandidate(null);
    expect(buf.candidate).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// Unit tests — render() returns a snapshot (Requirement 5.1, 5.3)
// ---------------------------------------------------------------------------

describe('OutputBuffer — render()', () => {
  it('render() returns the committed words and candidate', () => {
    const buf = new OutputBuffer();
    buf.accept('HELLO');
    buf.accept('WORLD');
    buf.setCandidate('TEST');

    const result = buf.render();

    expect(result.committed).toEqual(['HELLO', 'WORLD']);
    expect(result.candidate).toBe('TEST');
  });

  it('render() returns a copy — mutating the snapshot does not affect the buffer', () => {
    const buf = new OutputBuffer();
    buf.accept('HELLO');

    const result = buf.render();
    result.committed.push('MUTATED');

    // Internal state should be unaffected
    expect(buf.committed).toEqual(['HELLO']);
  });

  it('render() reflects the state after backspace()', () => {
    const buf = new OutputBuffer();
    buf.accept('HELLO');
    buf.accept('WORLD');
    buf.backspace();

    const result = buf.render();
    expect(result.committed).toEqual(['HELLO']);
  });
});
