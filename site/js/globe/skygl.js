// The SPHEREx all-sky HEALPix sheet on a sphere, drawn by WebGL2 (ported from Tracklet).

const TINT = [46, 30, 28];

const VERT = `#version 300 es
in vec2 pos;
out vec2 disc;
void main() { disc = pos; gl_Position = vec4(pos, 0.0, 1.0); }`;

const fragment = (g) => `#version 300 es
precision highp float;
precision highp int;
in vec2 disc;
out vec4 color;
uniform vec3 right, up, fwd;
uniform sampler2D sheet;
uniform float px;       // one output pixel, in disc units (radius = 1)
uniform int taps;       // taps per axis inside a pixel
uniform vec3 limb;      // the limb's exact centre (xy) and radius (z), in disc units

const int ORDER = ${g.order + Math.round(Math.log2(g.tile_px))};
const int NS = ${1 << (g.order + Math.round(Math.log2(g.tile_px)))};
const float NSF = float(NS);
const int TILE = ${g.tile_px};
const uint PER = uint(${g.tile_px * g.tile_px});
const uint NTILES = uint(${g.n_tiles});
const uint GW = uint(${g.grid_width});
const vec3 TINT = vec3(${TINT.map((v) => (v / 255).toFixed(6)).join(', ')});
const float HALF_PI = 1.57079632679;

uint spread(uint v) {
  v &= 0xffffu;
  v = (v | (v << 8)) & 0x00ff00ffu;
  v = (v | (v << 4)) & 0x0f0f0f0fu;
  v = (v | (v << 2)) & 0x33333333u;
  v = (v | (v << 1)) & 0x55555555u;
  return v;
}
uint compress(uint v) {
  v &= 0x55555555u;
  v = (v | (v >> 1)) & 0x33333333u;
  v = (v | (v >> 2)) & 0x0f0f0f0fu;
  v = (v | (v >> 4)) & 0x00ff00ffu;
  v = (v | (v >> 8)) & 0x0000ffffu;
  return v;
}

// healpix.js faceXy(), line for line.
vec3 cell(vec3 d) {
  float z = d.z, za = abs(z);
  float tt = mod(atan(d.y, d.x) / HALF_PI, 4.0);
  int face, ix, iy;
  if (za <= 2.0 / 3.0) {
    float t1 = NSF * (0.5 + tt), t2 = NSF * z * 0.75;
    int jp = int(floor(t1 - t2)), jm = int(floor(t1 + t2));
    int ifp = jp >> ORDER, ifm = jm >> ORDER;
    if (ifp == ifm) face = (ifp & 3) + 4;
    else if (ifp < ifm) face = ifp & 3;
    else face = (ifm & 3) + 8;
    ix = jm & (NS - 1);
    iy = NS - (jp & (NS - 1)) - 1;
  } else {
    int ntt = min(3, int(floor(tt)));
    float tp = tt - float(ntt);
    float tmp = NSF * sqrt(3.0 * (1.0 - za));
    int jp = min(int(floor(tp * tmp)), NS - 1);
    int jm = min(int(floor((1.0 - tp) * tmp)), NS - 1);
    if (z >= 0.0) { face = ntt; ix = NS - jm - 1; iy = NS - jp - 1; }
    else { face = ntt + 8; ix = jp; iy = jm; }
  }
  return vec3(float(face), float(ix), float(iy));
}

vec3 fetchSky(vec3 d) {
  vec3 c = cell(d);
  uint pix = uint(c.x) * uint(NS * NS) + spread(uint(c.y)) + 2u * spread(uint(c.z));
  uint sub = pix % PER, tile = pix / PER;
  if (tile >= NTILES) return TINT;
  ivec2 t = ivec2(int(tile % GW) * TILE + int(compress(sub)),
                  int(tile / GW) * TILE + TILE - 1 - int(compress(sub >> 1)));
  vec2 v = texelFetch(sheet, t, 0).rg;
  return v.g > 0.5 ? vec3(v.r) : TINT;
}

void main() {
  // Coverage of this pixel by the limb's circle, which is what the 2D clip
  // gave when the sky was drawn through one. The circle is the graticule's,
  // not the box's: the box is rounded to whole device pixels and can sit a
  // fraction of one off it.
  float cover = clamp(0.5 + (limb.z - length(disc - limb.xy)) / px, 0.0, 1.0);
  if (cover <= 0.0) discard;
  vec3 sum = vec3(0.0);
  float inv = 1.0 / float(taps);
  for (int j = 0; j < taps; j++) {
    for (int i = 0; i < taps; i++) {
      vec2 p = disc + ((vec2(float(i), float(j)) + 0.5) * inv - 0.5) * px;
      float q = dot(p, p);
      if (q > 1.0) { p *= inversesqrt(q); q = 1.0; }   // taps past the limb
      vec3 d = p.x * right + p.y * up + sqrt(1.0 - q) * fwd;
      sum += fetchSky(d);
    }
  }
  color = vec4(sum * inv * inv, 1.0) * cover;   // premultiplied
}`;

function compile(gl, type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) {
    throw new Error(gl.getShaderInfoLog(s) || 'shader failed to compile');
  }
  return s;
}

/** Resolves to `{ canvas, render(basis, box) -> bool, destroy() }`, or null
 *  when WebGL2 is not available and the caller should use its CPU path. The
 *  caller puts `canvas` on the page; destroy() takes it off again. */
export async function createSkyGL(sky) {
  const { geom, grey, alpha } = sky;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = 2;
  //: The buffer is not preserved: a WebGL canvas nothing draws into keeps
  //: showing its last frame, so a repaint at an unchanged orientation (a
  //: hover, a resize of something else) simply leaves it alone.
  const gl = canvas.getContext('webgl2', {
    alpha: true, premultipliedAlpha: true, antialias: false, depth: false,
    stencil: false, preserveDrawingBuffer: false,
  });
  if (!gl) return null;

  const prog = gl.createProgram();
  gl.attachShader(prog, compile(gl, gl.VERTEX_SHADER, VERT));
  gl.attachShader(prog, compile(gl, gl.FRAGMENT_SHADER, fragment(geom)));
  gl.linkProgram(prog);
  if (!gl.getProgramParameter(prog, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(prog));
  gl.useProgram(prog);

  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, 'pos');
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);

  // Two channels: grey, and whether the pixel was observed.
  const n = geom.image_width * geom.image_height;
  const rg = new Uint8Array(n * 2);
  for (let i = 0; i < n; i++) { rg[2 * i] = grey[i]; rg[2 * i + 1] = alpha[i] ? 255 : 0; }
  const tex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RG8, geom.image_width, geom.image_height, 0,
    gl.RG, gl.UNSIGNED_BYTE, rg);
  // texelFetch ignores filtering; NEAREST just keeps the texture complete.
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);

  const uRight = gl.getUniformLocation(prog, 'right');
  const uUp = gl.getUniformLocation(prog, 'up');
  const uFwd = gl.getUniformLocation(prog, 'fwd');
  const uPx = gl.getUniformLocation(prog, 'px');
  const uTaps = gl.getUniformLocation(prog, 'taps');
  const uLimb = gl.getUniformLocation(prog, 'limb');
  gl.uniform1i(gl.getUniformLocation(prog, 'sheet'), 0);
  gl.clearColor(0, 0, 0, 0);

  //: One source cell is 0.115 deg, 0.002 of the globe's radius. Taps per axis
  //: is how many it takes to cover the cells a pixel spans, capped: past four
  //: a pixel is averaging cells it can barely tell apart, and above one it is
  //: a small disc where every cell is already bigger than the pixel.
  const cellRadii = Math.sqrt(Math.PI / 3) / (1 << (geom.order + Math.round(Math.log2(geom.tile_px))));

  let last = '';
  return {
    canvas,
    /** Draw the near hemisphere into a canvas `width` x `height` device
     *  pixels, as a `side`-pixel disc whose box has its top-left corner at
     *  device pixel (`x`, `y`). The limb is the circle at (`cx`, `cy`),
     *  radius `r`, in device pixels. False if the context has been lost. */
    render(b, { x, y, side, width, height, cx, cy, r }) {
      if (gl.isContextLost()) return false;
      side = Math.max(2, side);
      const key = `${b.fwd}|${x},${y},${side}|${width}x${height}|${cx},${cy},${r}`;
      if (key === last && canvas.width === width && canvas.height === height) return true;
      if (canvas.width !== width || canvas.height !== height) {
        canvas.width = width;
        canvas.height = height;
      }
      const half = side / 2;
      const pxSize = 2 / side;
      gl.clear(gl.COLOR_BUFFER_BIT);   // the whole canvas: clear ignores the viewport
      // GL counts rows from the bottom.
      gl.viewport(x, height - y - side, side, side);
      gl.uniform3fv(uRight, b.right);
      gl.uniform3fv(uUp, b.up);
      gl.uniform3fv(uFwd, b.fwd);
      gl.uniform1f(uPx, pxSize);
      gl.uniform1i(uTaps, Math.max(2, Math.min(4, Math.ceil((pxSize / cellRadii) * 1.5))));
      gl.uniform3f(uLimb, (cx - x - half) / half, (y + half - cy) / half, r / half);
      gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
      last = key;
      return true;
    },
    destroy() {
      gl.getExtension('WEBGL_lose_context')?.loseContext();
      canvas.remove();
    },
  };
}
