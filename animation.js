const canvas = document.querySelector('#facet-canvas');
const wordmark = canvas.parentElement;
const motionButton = document.querySelector('#hero-motion');
const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const context = canvas.getContext('2d');

const sceneWidth = 520;
const sceneHeight = 184;
const cycle = 11;
const fontSize = 124;
let wordmarkStyle;
let splats;

function readWordmarkStyle(style, text) {
  const font = style.fontWeight + ' ' + fontSize + 'px ' + style.fontFamily;
  const spacing = fontSize * parseFloat(style.letterSpacing) / parseFloat(style.fontSize);
  context.font = font;
  const metrics = context.measureText(text);
  const glyphs = Array.from(text, (character, index) => ({
    character,
    offset: context.measureText(text.slice(0, index + 1)).width -
      context.measureText(character).width + spacing * index,
  }));
  return {
    font,
    glyphs,
    width: metrics.width + spacing * (glyphs.length - 1),
    baseline: (metrics.actualBoundingBoxAscent - metrics.actualBoundingBoxDescent) / 2,
  };
}

function paintTextMask(target) {
  const { font, glyphs, width, baseline } = wordmarkStyle;
  target.font = font;
  target.fillStyle = '#fff';
  for (const { character, offset } of glyphs) {
    target.fillText(character, offset - width / 2, baseline);
  }
}

function rasterSurface(width, height) {
  const surface = document.createElement('canvas');
  surface.width = width;
  surface.height = height;
  return surface;
}

function splatSprite(rgb) {
  const size = 64;
  const surface = rasterSurface(size, size);
  const paint = surface.getContext('2d');
  const center = size / 2;
  const gradient = paint.createRadialGradient(center, center, 0, center, center, center - 1);
  gradient.addColorStop(0, 'rgba(' + rgb + ', .94)');
  gradient.addColorStop(.51, 'rgba(' + rgb + ', .75)');
  gradient.addColorStop(.86, 'rgba(' + rgb + ', .37)');
  gradient.addColorStop(1, 'rgba(' + rgb + ', 0)');
  paint.fillStyle = gradient;
  paint.fillRect(0, 0, size, size);
  paint.beginPath();
  paint.arc(center, center, center * .89, 0, 2 * Math.PI);
  paint.strokeStyle = 'rgba(255, 246, 230, .59)';
  paint.lineWidth = 2.3;
  paint.stroke();
  return surface;
}

function makeSplats() {
  const mask = rasterSurface(sceneWidth, sceneHeight);
  const maskContext = mask.getContext('2d');
  maskContext.translate(sceneWidth / 2, sceneHeight / 2);
  paintTextMask(maskContext);
  const pixels = maskContext.getImageData(0, 0, sceneWidth, sceneHeight).data;
  const sprites = ['39, 71, 91', '59, 100, 118', '90, 119, 129', '149, 92, 69', '191, 111, 71', '218, 144, 90'].map(splatSprite);

  let seed = 2951;
  function random() {
    seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0;
    return seed / 4294967296;
  }

  function covered(x, y) {
    return pixels[(Math.round(y) * sceneWidth + Math.round(x)) * 4 + 3] > 160;
  }

  const points = [];
  const stride = 4.6;
  for (let row = 7; row < sceneHeight - 7; row += stride) {
    const stagger = Math.round(row / stride) % 2 ? 0 : stride / 2;
    for (let column = 7 + stagger; column < sceneWidth - 7; column += stride) {
      const x = column + (random() - .5) * 2.1;
      const y = row + (random() - .5) * 2.1;
      if (!covered(x, y)) continue;

      const neighbors = [[-2, 0], [2, 0], [0, -2], [0, 2]];
      const edge = neighbors.some(([dx, dy]) => !covered(x + dx, y + dy));
      const radius = (edge ? 2.9 : 4.6) + random() * 1.9;
      const palettePosition = (y - sceneHeight / 2 + 43) / 87 + (random() - .5) * .24;
      const paletteIndex = Math.max(0, Math.min(sprites.length - 1, Math.floor(palettePosition * sprites.length)));
      points.push({
        x: x - sceneWidth / 2,
        y: y - sceneHeight / 2,
        radius,
        aspect: .71 + random() * .28,
        angle: (random() - .5) * 2.4,
        tilt: (random() - .5) * 1.42,
        depth: 14 + Math.floor(random() * 5) * 15 + random() * 9,
        sprite: sprites[paletteIndex],
      });
    }
  }
  // Match the depth order of the first lifted frame while the splats are still coplanar.
  return points.sort((first, second) => first.depth - second.depth);
}

function smoothstep(from, to, time) {
  const fraction = Math.min(1, Math.max(0, (time - from) / (to - from)));
  return fraction * fraction * (3 - 2 * fraction);
}

function paintSplats(project, lift, scale) {
  const visibleSplats = splats.map(splat => {
    const { x, y, depth } = splat;
    const spread = lift * depth;
    const center = [x + spread * Math.sin(x * .024) * .10, y + spread * Math.sin(x * .043) * .12, spread];
    return { splat, center, screen: project(...center) };
  }).sort((first, second) => first.screen[2] - second.screen[2]);

  for (const { splat, center, screen: [x, y] } of visibleSplats) {
    const { angle, tilt, radius, aspect, sprite } = splat;
    const ax = Math.cos(angle) * radius;
    const ay = Math.sin(angle) * radius;
    const az = radius * lift * Math.sin(tilt);
    const [uX, uY] = project(center[0] + ax, center[1] + ay, center[2] + az);
    const [vX, vY] = project(center[0] - ay * aspect, center[1] + ax * aspect, center[2] - az * .4);
    context.setTransform(
      (uX - x) * scale, (uY - y) * scale, (vX - x) * scale, (vY - y) * scale,
      x * scale + canvas.width / 2, y * scale + canvas.height / 2
    );
    context.drawImage(sprite, -1, -1, 2, 2);
  }
}

function drawWordmark(time) {
  const lift = smoothstep(2.65, 5.35, time) * (1 - smoothstep(6.65, 8.85, time));
  const cameraTurn = smoothstep(2.95, 5.75, time) * (1 - smoothstep(6.6, 8.7, time));
  const yaw = -cameraTurn * .46;
  const pitch = cameraTurn * .33;
  const cosY = Math.cos(yaw), sinY = Math.sin(yaw);
  const cosX = Math.cos(pitch), sinX = Math.sin(pitch);

  function project(x, y, z) {
    const turnedX = x * cosY + z * sinY;
    const turnedZ = z * cosY - x * sinY;
    const tiltedY = y * cosX - turnedZ * sinX;
    const depth = y * sinX + turnedZ * cosX;
    const perspective = 1050 / (1050 - depth);
    return [turnedX * perspective + lift * 23, tiltedY * perspective + lift * 14, depth];
  }

  const scale = Math.min(canvas.width / sceneWidth, canvas.height / sceneHeight);
  context.setTransform(1, 0, 0, 1, 0, 0);
  context.clearRect(0, 0, canvas.width, canvas.height);
  paintSplats(project, lift, scale);
}

if (context) {
  const typography = getComputedStyle(wordmark);
  wordmarkStyle = readWordmarkStyle(typography, wordmark.getAttribute('aria-label'));
  splats = makeSplats();
  let playing = !reducedMotion.matches;
  let visible = true;
  let frame;
  let previousTime;
  let elapsed = 0;

  function resize() {
    const ratio = Math.min(devicePixelRatio || 1, 2);
    canvas.width = Math.round(wordmark.clientWidth * ratio);
    canvas.height = Math.round(wordmark.clientHeight * ratio);
    if (canvas.width) drawWordmark(elapsed);
  }

  function tick(time) {
    if (previousTime !== undefined) elapsed = (elapsed + Math.min(.05, (time - previousTime) / 1000)) % cycle;
    previousTime = time;
    drawWordmark(elapsed);
    frame = requestAnimationFrame(tick);
  }

  function updateMotion() {
    motionButton.textContent = playing ? 'Pause animation' : 'Play animation';
    if (playing && visible && !document.hidden) {
      if (frame === undefined) frame = requestAnimationFrame(tick);
    } else if (frame !== undefined) {
      cancelAnimationFrame(frame);
      frame = undefined;
      previousTime = undefined;
    }
  }

  motionButton.addEventListener('click', () => {
    playing = !playing;
    updateMotion();
  });
  reducedMotion.addEventListener('change', () => {
    playing = !reducedMotion.matches;
    if (!playing) {
      elapsed = 0;
      drawWordmark(elapsed);
    }
    updateMotion();
  });
  document.addEventListener('visibilitychange', updateMotion);
  new ResizeObserver(resize).observe(wordmark);
  new IntersectionObserver(([entry]) => {
    visible = entry.isIntersecting;
    updateMotion();
  }).observe(wordmark);

  resize();
  wordmark.classList.add('is-rendered');
  motionButton.hidden = false;
  updateMotion();
}
