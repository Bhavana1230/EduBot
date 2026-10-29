const messagesEl = document.getElementById('messages');
const form = document.getElementById('chatForm');
const input = document.getElementById('msgInput');
const langPref = document.getElementById('langPref');

function addMsg(text, who, escalate) {
  const div = document.createElement('div');
  div.className = 'msg ' + who + (escalate ? ' escalate' : '');
  div.textContent = text;
  messagesEl.appendChild(div);
  messagesEl.scrollTop = messagesEl.scrollHeight;
}

function chip(text, cls) {
  const s = document.createElement('span');
  s.className = 'chip ' + (cls || '');
  s.textContent = text;
  return s;
}

function updateInsights(d) {
  document.getElementById('ivLang').textContent = ({en: 'English', hi: 'Hindi', mr: 'Marathi', hinglish: 'Hinglish'})[d.language] || d.language;
  document.getElementById('ivIntent').textContent = d.intent;
  document.getElementById('ivConf').textContent = 'confidence: ' + (d.confidence * 100).toFixed(1) + '%';
  const top3 = document.getElementById('ivTop3'); top3.innerHTML = '';
  d.top3.forEach((t, i) => top3.appendChild(chip(t[0] + ' ' + (t[1] * 100).toFixed(0) + '%', i === 0 ? 'top1' : '')));
  const pos = document.getElementById('ivPos'); pos.innerHTML = '';
  d.pos.forEach(([tok, tag]) => pos.appendChild(chip(tok + '/' + tag, 'pos-tag')));
  const ents = document.getElementById('ivEntities'); ents.innerHTML = '';
  const e = d.entities;
  if (Object.keys(e).length === 0) { ents.appendChild(chip('none detected')); }
  Object.entries(e).forEach(([k, v]) => ents.appendChild(chip(k + ': ' + (Array.isArray(v) ? v.join(',') : v), 'entity')));
  document.getElementById('ivSentiment').textContent = d.sentiment.label + ' (' + d.sentiment.score + ')';
  document.getElementById('ivSource').textContent = d.source;
}

async function send(text) {
  addMsg(text, 'user');
  input.value = '';
  const res = await fetch('/api/chat', {
    method: 'POST', headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({message: text, lang_pref: langPref.value})
  });
  const data = await res.json();
  if (data.error) { addMsg('Something went wrong. Please try again.', 'bot'); return; }
  addMsg(data.reply, 'bot', data.escalate);
  updateInsights(data);
}

form.addEventListener('submit', (e) => { e.preventDefault(); const t = input.value.trim(); if (t) send(t); });
document.querySelectorAll('.quick-chips button').forEach(b => b.addEventListener('click', () => send(b.dataset.q)));
document.getElementById('resetBtn').addEventListener('click', async () => {
  await fetch('/api/reset', {method: 'POST'});
  messagesEl.innerHTML = '';
  addMsg("Hello! I'm EduBot. Ask me about attendance, timetable, exams, assignments, admissions, fees and more.", 'bot');
});

addMsg("Hello! I'm EduBot, your student query assistant. Ask me about attendance, timetable, exams, assignments, admissions, fees and more.", 'bot');
