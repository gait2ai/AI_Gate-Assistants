// static/sw.js

const CACHE_NAME_PREFIX = 'ai-gate-assistant-cache';
const CACHE_VERSION = 'v1.4'; // Increment this version when APP_SHELL_URLS or critical assets change
const CACHE_NAME = `${CACHE_NAME_PREFIX}-${CACHE_VERSION}`;

// Core application assets (App Shell) to be cached on install
const APP_SHELL_URLS = [
  '/', // Main page (index.html)
  'manifest.json',
  'css/style.css',
  'js/app.js',
  'assets/logo.png',
  'assets/logo_192.png', // For PWA manifest
  'assets/logo_512.png', // For PWA manifest
  'assets/favicon.ico',
  'offline.html' // Offline fallback page
];

// API endpoints whose GET responses might be cached
const API_CACHE_URLS = [
  '/api/institution',
  '/health'
];

// --- Service Worker Lifecycle ---

self.addEventListener('install', (event) => {
  console.log(`[Service Worker] Installing ${CACHE_NAME}...`);
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('[Service Worker] Caching App Shell URLs...');
        return cache.addAll(APP_SHELL_URLS);
      })
      .then(() => {
        console.log(`[Service Worker] ${CACHE_NAME} installed successfully.`);
        // If you want the new service worker to take control immediately
        // return self.skipWaiting();
      })
      .catch(error => {
        console.error('[Service Worker] Installation failed:', error);
      })
  );
});

self.addEventListener('activate', (event) => {
  console.log(`[Service Worker] Activating ${CACHE_NAME}...`);
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames.map((cacheName) => {
          if (cacheName.startsWith(CACHE_NAME_PREFIX) && cacheName !== CACHE_NAME) {
            console.log(`[Service Worker] Deleting old cache: ${cacheName}`);
            return caches.delete(cacheName);
          }
        })
      );
    }).then(() => {
      console.log(`[Service Worker] ${CACHE_NAME} activated and old caches cleaned.`);
      return self.clients.claim(); // Take control of open clients immediately
    })
  );
});

self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // Strategy 1: Cache, Falling Back to Network (for App Shell & Static Assets)
  if (APP_SHELL_URLS.includes(url.pathname) ||
      request.destination === 'style' ||
      request.destination === 'script' ||
      request.destination === 'image' ||
      request.destination === 'font' ||
      request.destination === 'manifest') {
    event.respondWith(
      caches.match(request)
        .then((cachedResponse) => {
          if (cachedResponse) {
            return cachedResponse;
          }
          return fetch(request).then((networkResponse) => {
            if (!networkResponse || networkResponse.status !== 200 || networkResponse.type !== 'basic') {
              // Do not cache opaque or bad responses for the app shell
              return networkResponse;
            }
            const responseToCache = networkResponse.clone();
            caches.open(CACHE_NAME)
              .then((cache) => {
                cache.put(request, responseToCache);
              });
            return networkResponse;
          }).catch(error => {
            console.error(`[Service Worker] Fetch failed for asset ${request.url}:`, error);
            if (request.mode === 'navigate') { // If it's a navigation request (e.g., to an HTML page)
              return caches.match('offline.html');
            }
            // For other assets, let the browser handle the error (e.g., show broken image)
            // Or throw error to signify failure if specific handling is needed upstream.
          });
        })
    );
    return;
  }

  // Strategy 2: Network, Falling Back to Cache (for API GET requests)
  if (request.method === 'GET' && API_CACHE_URLS.includes(url.pathname)) {
    event.respondWith(
      fetch(request)
        .then((networkResponse) => {
          if (!networkResponse || networkResponse.status !== 200) {
            // Network request failed or returned an error, try cache
            return caches.match(request).then(cachedResponse => {
              return cachedResponse || networkResponse; // Return network error if no cache
            });
          }
          // Network request was successful, cache it
          const responseToCache = networkResponse.clone();
          caches.open(CACHE_NAME)
            .then((cache) => {
              cache.put(request, responseToCache);
            });
          return networkResponse;
        })
        .catch(() => {
          // Network completely failed (e.g., offline)
          return caches.match(request).then((cachedResponse) => {
            if (cachedResponse) {
              return cachedResponse;
            }
            // No cache and no network, return a standard error response
            return new Response(JSON.stringify({ error: 'Offline and no cache available for this API endpoint.' }), {
              headers: { 'Content-Type': 'application/json' },
              status: 503,
              statusText: 'Service Unavailable - Offline'
            });
          });
        })
    );
    return;
  }

  // For all other requests (e.g., POST to /api/chat), let them pass through to the network.
  // This is the default browser behavior if event.respondWith() is not called.
});

// Optional: Listener for messages from the client (e.g., to trigger self.skipWaiting())
// self.addEventListener('message', (event) => {
//   if (event.data && event.data.type === 'SKIP_WAITING') {
//     console.log('[Service Worker] SKIP_WAITING message received, skipping waiting...');
//     self.skipWaiting();
//   }
// });