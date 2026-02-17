// Prevent indexing by search engines
const meta = document.createElement('meta');
meta.name = "robots";
meta.content = "noindex, nofollow";
document.getElementsByTagName('head')[0].appendChild(meta);
