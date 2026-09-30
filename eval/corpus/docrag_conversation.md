# Conversation memory

## Follow-up questions

The server keeps no conversation state. The client sends the most recent turns of the
conversation with each question. When a follow-up question depends on them, the server
first condenses it into a standalone retrieval query, then runs the search with that
query instead of the raw follow-up.

## History limits

History is capped by message count and by characters, so a long conversation cannot
crowd the retrieved context out of the model's window. Stopped or failed answers are not
sent back as history.
