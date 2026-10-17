/* Opens the existing Ask dock with a question already sent. Uses only the
 * dock's own controls and its existing /api/chat path. */
export function askAbout(question) {
  const chat = document.getElementById('chat');
  if (chat.hidden) document.getElementById('chat-btn').click();
  const input = document.getElementById('chat-input');
  input.value = question;
  document.getElementById('chat-form').requestSubmit();
  input.focus();
}
