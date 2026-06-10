/**
 * GlossPostProcessor — in-browser gloss-to-English translation.
 *
 * Runs T5-small (INT8 quantized, ~80 MB) via Transformers.js to convert a
 * sequence of ASL gloss tokens into grammatically natural English text.
 *
 * Loaded as a background task after the ONNX classifier is ready (staged
 * loading). Uses a dynamic import inside `load()` so the module is safe to
 * import in Node.js test environments that lack WebGPU / browser APIs.
 *
 * Requirements: 3.4, 5.9
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * Minimal shape of the translation pipeline returned by Transformers.js.
 * We only call it and read the first element's `translation_text`.
 */
interface TranslationOutput {
  translation_text: string;
}

/**
 * Minimal pipeline callable type. The actual type from @huggingface/transformers
 * is complex; we narrow to what we actually use.
 */
type TranslationPipeline = (
  input: string,
  options?: Record<string, unknown>,
) => Promise<TranslationOutput | TranslationOutput[]>;

// ---------------------------------------------------------------------------
// GlossPostProcessor
// ---------------------------------------------------------------------------

/**
 * Converts a sequence of ASL glosses into grammatically natural English text
 * using a quantized T5-small model loaded via Transformers.js.
 *
 * Usage:
 *   const processor = new GlossPostProcessor();
 *   await processor.load();                   // background load after ONNX ready
 *   const text = await processor.glossToEnglish(['WANT', 'EAT', 'PIZZA']);
 *   // → "I want to eat pizza."
 */
export class GlossPostProcessor {
  private readonly modelId: string;
  private readonly dtype: string;

  /** The loaded translation pipeline, or null if not yet loaded. */
  private pipeline: TranslationPipeline | null = null;

  /** Whether the T5 model has been loaded successfully. */
  public isLoaded: boolean = false;

  /**
   * @param modelId  Hugging Face model identifier (default: "Xenova/t5-small").
   * @param dtype    Quantization dtype passed to the pipeline (default: "q8").
   *                 The value is forwarded as-is to Transformers.js; we use
   *                 `as unknown` at the call site to avoid coupling to the
   *                 library's internal dtype union.
   */
  constructor(
    modelId = 'Xenova/t5-small',
    dtype = 'q8',
  ) {
    this.modelId = modelId;
    this.dtype = dtype;
  }

  /**
   * Load the T5 translation pipeline via a dynamic import of
   * `@huggingface/transformers`.
   *
   * The dynamic import is intentional: it keeps this module safe to `import`
   * in Node.js test environments that lack WebGPU / ONNX Runtime Web. The
   * actual model bytes are only fetched when `load()` is called at runtime.
   *
   * Errors (e.g. network failure, missing WebGPU) are caught and logged; in
   * that case `isLoaded` remains false and `glossToEnglish` falls back to raw
   * gloss output.
   */
  async load(): Promise<void> {
    try {
      // Dynamic import keeps the module tree safe in non-browser environments.
      const { pipeline } = await import('@huggingface/transformers');

      // Cast to our minimal interface; dtype is a valid option accepted by
      // Transformers.js translation pipelines.
      // `this.dtype` is a plain string; we cast through `unknown` to avoid
      // coupling to the exact dtype union exported by the library version.
      this.pipeline = (await pipeline('translation', this.modelId, {
        dtype: this.dtype as unknown as 'q8',
      })) as unknown as TranslationPipeline;

      this.isLoaded = true;
    } catch (err) {
      this.isLoaded = false;
      console.warn(
        '[GlossPostProcessor] Failed to load T5 model — ' +
        'falling back to raw gloss output.',
        err,
      );
    }
  }

  /**
   * Translate a sequence of ASL gloss tokens into natural English text.
   *
   * Behaviour table:
   * | glosses         | model loaded | result                          |
   * |-----------------|-------------|----------------------------------|
   * | []              | any          | "" (empty string)               |
   * | non-empty       | false        | glosses joined with spaces       |
   * | non-empty       | true         | T5 translation (or fallback)     |
   *
   * The fallback guarantees that a non-empty input always yields a non-empty
   * output, satisfying Property 5.
   *
   * Requirements: 3.4, 5.9
   *
   * @param glosses  Array of uppercase ASL gloss tokens, e.g. ['WANT', 'EAT'].
   * @returns        Grammatically natural English string, or raw glosses joined
   *                 with spaces if the model is unavailable or returns empty.
   */
  async glossToEnglish(glosses: string[]): Promise<string> {
    // Empty input → empty output (defined behaviour).
    if (glosses.length === 0) {
      return '';
    }

    const rawFallback = glosses.join(' ');

    // Model not loaded → raw fallback (always non-empty for non-empty input).
    if (!this.isLoaded || this.pipeline === null) {
      return rawFallback;
    }

    try {
      const input = 'translate ASL to English: ' + rawFallback;
      const output = await this.pipeline(input);

      // Unwrap array or object result.
      const result: TranslationOutput = Array.isArray(output) ? output[0] : output;
      const translation = result?.translation_text?.trim() ?? '';

      // Guard: never return empty for non-empty input.
      return translation.length > 0 ? translation : rawFallback;
    } catch (err) {
      console.warn('[GlossPostProcessor] Translation failed — using raw gloss fallback.', err);
      return rawFallback;
    }
  }
}

// ---------------------------------------------------------------------------
// Singleton export
// ---------------------------------------------------------------------------

/**
 * Shared singleton instance of GlossPostProcessor.
 *
 * Call `glossProcessor.load()` once after the ONNX classifier is ready.
 * Then call `glossProcessor.glossToEnglish(glosses)` whenever a committed
 * gloss buffer needs to be translated.
 */
export const glossProcessor = new GlossPostProcessor();
