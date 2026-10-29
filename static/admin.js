async function loadStats() {
  const res = await fetch('/api/stats');
  const d = await res.json();
  const grid = document.getElementById('statGrid');
  grid.innerHTML = '';
  const cards = [
    ['Total queries', d.total], ['Avg. confidence', (d.avg_confidence * 100).toFixed(1) + '%'],
    ['Escalations flagged', d.escalations], ['Distinct intents', Object.keys(d.intents).length]
  ];
  cards.forEach(([lbl, num]) => {
    const c = document.createElement('div'); c.className = 'card';
    c.innerHTML = '<div class="num">' + num + '</div><div class="lbl">' + lbl + '</div>';
    grid.appendChild(c);
  });
  const ic = document.getElementById('intentChart').getContext('2d');
  new Chart(ic, {type: 'bar', data: {labels: Object.keys(d.intents), datasets: [{label: 'Queries', data: Object.values(d.intents), backgroundColor: '#6c8cff'}]},
    options: {plugins: {legend: {display: false}}, scales: {x: {ticks: {color: '#8b93ad'}}, y: {ticks: {color: '#8b93ad'}}}}});
  const sc = document.getElementById('sentimentChart').getContext('2d');
  new Chart(sc, {type: 'doughnut', data: {labels: Object.keys(d.sentiment), datasets: [{data: Object.values(d.sentiment), backgroundColor: ['#35d0a8', '#8b93ad', '#ff6b7a']}]},
    options: {plugins: {legend: {labels: {color: '#e8ecf7'}}}}});
}
loadStats();

document.getElementById('faqForm').addEventListener('submit', async (e) => {
  e.preventDefault();
  const q = document.getElementById('faqQ').value.trim();
  const a = document.getElementById('faqA').value.trim();
  const intent = document.getElementById('faqIntent').value.trim() || 'custom';
  const res = await fetch('/api/add_faq', {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({question: q, answer: a, intent})});
  const d = await res.json();
  document.getElementById('faqMsg').textContent = d.ok ? ('Added. Knowledge base now has ' + d.count + ' custom entries.') : (d.error || 'Failed');
  if (d.ok) { document.getElementById('faqForm').reset(); }
});
