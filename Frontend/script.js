async function sendQuery() {
  const query = document.getElementById("queryInput").value;
  const queryType = document.getElementById("queryType").value;
  const chatWindow = document.getElementById("chatWindow");

  if (!query.trim()) {
    alert("Please enter a question");
    return;
  }

  // Add user bubble
  chatWindow.innerHTML += `<div class="message user"><b>You:</b> ${query}</div>`;

  const res = await fetch("http://127.0.0.1:5000/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, query_type: queryType })
  });

  const data = await res.json();

  // Add bot bubble
  chatWindow.innerHTML += `<div class="message bot"><b>Lia:</b> ${data.answer}</div>`;

  document.getElementById("queryInput").value = "";
  chatWindow.scrollTop = chatWindow.scrollHeight;
}
