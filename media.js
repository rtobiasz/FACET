// The method video shows a local poster and loads the YouTube player only on click.
// A file:// preview cannot identify the embedding page to YouTube, so it keeps the plain link.
const videoFrame = document.querySelector('#project-video .video-frame');
if (videoFrame && (location.protocol === 'http:' || location.protocol === 'https:')) {
  videoFrame.querySelector('.video-poster').addEventListener('click', event => {
    event.preventDefault();
    const player = document.createElement('iframe');
    player.src = 'https://www.youtube-nocookie.com/embed/' + videoFrame.dataset.video + '?autoplay=1&rel=0';
    player.title = 'FACET method video';
    player.allow = 'accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; fullscreen';
    player.referrerPolicy = 'strict-origin-when-cross-origin';
    player.allowFullscreen = true;
    videoFrame.replaceChildren(player);
    player.focus();
  });
}

const copyButton = document.querySelector('.copy-bibtex');
if (copyButton && navigator.clipboard) {
  copyButton.hidden = false;
  copyButton.addEventListener('click', async () => {
    try {
      await navigator.clipboard.writeText(document.querySelector('#bibtex-code').textContent);
      copyButton.textContent = 'Copied';
    } catch {
      copyButton.textContent = 'Copy failed';
    }
    setTimeout(() => { copyButton.textContent = 'Copy'; }, 2000);
  });
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
