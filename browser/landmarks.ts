/**
 * Landmark normalization utilities for the browser inference path.
 *
 * Mirrors the Python logic in utils/landmarks.py exactly:
 *   1. Translate so the wrist (index 0) is at the origin.
 *   2. Scale by the wrist → index-MCP (index 5) distance.
 *   3. For left-hand landmarks, mirror the x-axis.
 *
 * Requirements: 1.3, 4.1
 */

/** A single 3-D landmark [x, y, z]. */
export type Point3D = [number, number, number];

/** Raw hand landmarks from MediaPipe: 21 keypoints. */
export type HandLandmarks = Point3D[];

const WRIST_IDX = 0;
const INDEX_MCP_IDX = 5;
const NUM_LANDMARKS = 21;
const SINGLE_HAND_DIM = 63; // 21 × 3

/**
 * Normalize a single hand's 21 landmarks to a 63-float vector.
 *
 * @param handLm    Array of 21 [x,y,z] points from MediaPipe.
 * @param mirrorLeft When true, flip x so left-hand poses align with right-hand.
 * @returns Float32Array of length 63, or null if the input is degenerate.
 */
export function normalizeLandmarks(
  handLm: HandLandmarks,
  mirrorLeft = false,
): Float32Array | null {
  if (!handLm || handLm.length !== NUM_LANDMARKS) return null;

  // Copy into a flat buffer [x0,y0,z0, x1,y1,z1, ...]
  const lm = new Float32Array(NUM_LANDMARKS * 3);
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    lm[i * 3 + 0] = handLm[i][0];
    lm[i * 3 + 1] = handLm[i][1];
    lm[i * 3 + 2] = handLm[i][2];
  }

  // 1. Translate to wrist origin
  const wx = lm[WRIST_IDX * 3 + 0];
  const wy = lm[WRIST_IDX * 3 + 1];
  const wz = lm[WRIST_IDX * 3 + 2];
  for (let i = 0; i < NUM_LANDMARKS; i++) {
    lm[i * 3 + 0] -= wx;
    lm[i * 3 + 1] -= wy;
    lm[i * 3 + 2] -= wz;
  }

  // 2. Scale by wrist → index-MCP distance
  const ix = lm[INDEX_MCP_IDX * 3 + 0];
  const iy = lm[INDEX_MCP_IDX * 3 + 1];
  const iz = lm[INDEX_MCP_IDX * 3 + 2];
  const scale = Math.sqrt(ix * ix + iy * iy + iz * iz);
  if (scale < 1e-6) return null;

  for (let i = 0; i < NUM_LANDMARKS * 3; i++) {
    lm[i] /= scale;
  }

  // 3. Mirror left-hand x-axis
  if (mirrorLeft) {
    for (let i = 0; i < NUM_LANDMARKS; i++) {
      lm[i * 3 + 0] *= -1;
    }
  }

  return lm;
}

/**
 * Combine right and left hand normalized landmarks into a 126-float vector.
 *
 * Missing hands are represented as all-zeros.  Left hand is mirrored so both
 * hands share the same coordinate convention.
 *
 * @param rightLm Normalized right-hand landmarks (21 points), or null.
 * @param leftLm  Normalized left-hand landmarks (21 points), or null.
 * @returns Float32Array of length 126: [right_63 | left_63].
 */
export function buildTwoHandVector(
  rightLm: HandLandmarks | null,
  leftLm: HandLandmarks | null,
): Float32Array {
  const out = new Float32Array(126); // zero-filled by default

  if (rightLm !== null) {
    const vec = normalizeLandmarks(rightLm, false);
    if (vec !== null) out.set(vec, 0);
  }

  if (leftLm !== null) {
    const vec = normalizeLandmarks(leftLm, true);
    if (vec !== null) out.set(vec, SINGLE_HAND_DIM);
  }

  return out;
}
