import { afterEach, describe, expect, it } from 'vitest';
import type { Conversation } from '../../src/hooks/chat/model';
import { addManyToTrash, addToTrash, readTrash, trashRoom, TRASH_MAX_CONVERSATIONS } from '../../src/utils/trash';

function conversation(id: string): Conversation {
  return {
    id,
    title: id,
    createdAt: 1,
    updatedAt: 1,
    turns: [{ id: `${id}-q`, role: 'user', content: 'Q', sources: [], timestamp: 1, timings: null, traceId: null }],
  };
}

afterEach(() => localStorage.clear());

describe('pass 21: removals made to import more never push out earlier deletes', () => {
  it('fills only the room left, in one write', () => {
    for (let i = 0; i < 15; i += 1) addToTrash(conversation(`mine-${i}`), Date.now() - 1000);
    expect(trashRoom()).toBe(TRASH_MAX_CONVERSATIONS - 15);

    const added = addManyToTrash(
      Array.from({ length: 40 }, (_, i) => conversation(`imported-room-${i}`)),
    );

    expect(added).toBe(5);
    const ids = readTrash().map((entry) => entry.conversation.id);
    expect(ids).toHaveLength(TRASH_MAX_CONVERSATIONS);
    for (let i = 0; i < 15; i += 1) expect(ids).toContain(`mine-${i}`);
    expect(trashRoom()).toBe(0);
    expect(addManyToTrash([conversation('one-more')])).toBe(0);
  });
});
