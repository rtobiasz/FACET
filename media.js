// A file:// preview cannot identify the embedding page to YouTube.
if (location.protocol === 'http:' || location.protocol === 'https:') {
  const player = document.querySelector('#project-video iframe[data-src]');
  player.src = player.dataset.src;
  player.hidden = false;
}

const prefersReducedMotion = matchMedia('(prefers-reduced-motion: reduce)');
const playbackUpdates = new Map();
const visibilityObserver = new IntersectionObserver(entries => {
  for (const entry of entries) {
    const group = playbackUpdates.get(entry.target);
    group.inView = entry.isIntersecting;
    group.update();
  }
}, { threshold: 0.35 });

for (const card of document.querySelectorAll('.gallery .clip-group')) {
  const button = card.querySelector('.clip-toggle');
  const videos = Array.from(card.querySelectorAll('video'));
  const [leader, ...followers] = videos;
  const title = card.querySelector('h4').textContent;
  const gallery = card.closest('details');
  let preference;
  let playing = false;

  function updateButton() {
    const action = playing ? 'Pause' : 'Play';
    button.textContent = action;
    button.setAttribute('aria-label', action + ' ' + title);
  }

  function synchronize() {
    for (const follower of followers) {
      if (follower.readyState > 0 && Math.abs(follower.currentTime - leader.currentTime) > .10) {
        follower.currentTime = leader.currentTime;
      }
    }
  }

  function update() {
    const shouldPlay = gallery.open && !document.hidden &&
      (preference === true || (group.inView && preference !== false && !prefersReducedMotion.matches));
    if (playing === shouldPlay) return;

    playing = shouldPlay;
    updateButton();
    if (shouldPlay) {
      synchronize();
      Promise.all(videos.map(video => video.play())).catch(error => {
        if (!playing || error.name === 'AbortError') return;
        preference = false;
        update();
        for (const video of videos) video.controls = true;
      });
    } else {
      for (const video of videos) video.pause();
    }
  }

  const group = { inView: false, update() {
    if (!this.inView && preference === true) preference = undefined;
    update();
  } };
  playbackUpdates.set(card, group);
  for (const video of videos) video.controls = false;
  updateButton();
  button.hidden = false;
  button.addEventListener('click', () => {
    preference = !playing;
    update();
  });
  leader.addEventListener('timeupdate', synchronize);
  gallery.addEventListener('toggle', () => {
    if (!gallery.open && preference === true) preference = undefined;
    update();
  });
  visibilityObserver.observe(card);
}

function updatePlayback() {
  for (const group of playbackUpdates.values()) group.update();
}

document.addEventListener('visibilitychange', updatePlayback);
prefersReducedMotion.addEventListener('change', updatePlayback);
