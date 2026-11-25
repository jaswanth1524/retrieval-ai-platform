/** Saved questions to start from ("What does {document} say about {topic}?"), kept in
 *  this browser: inserted from the composer's Templates list, the first ``{…}``
 *  placeholder selected to type over. */
import { MAX_QUESTION_CHARS } from './limits';

export const TEMPLATES_STORAGE_KEY = 'docrag-question-templates';
export const MAX_TEMPLATES = 50;
export const MAX_TEMPLATE_NAME_CHARS = 60;

export interface QuestionTemplate {
  name: string;
  text: string;
}

export function readTemplates(): QuestionTemplate[] {
  let raw: unknown;
  try {
    raw = JSON.parse(localStorage.getItem(TEMPLATES_STORAGE_KEY) ?? '[]');
  } catch {
    return [];
  }
  if (!Array.isArray(raw)) return [];
  return raw
    .filter(
      (item): item is QuestionTemplate =>
        !!item &&
        typeof (item as QuestionTemplate).name === 'string' &&
        typeof (item as QuestionTemplate).text === 'string',
    )
    .map((item) => ({
      name: item.name.slice(0, MAX_TEMPLATE_NAME_CHARS),
      text: item.text.slice(0, MAX_QUESTION_CHARS),
    }))
    .slice(0, MAX_TEMPLATES);
}

function writeTemplates(templates: QuestionTemplate[]): boolean {
  try {
    localStorage.setItem(TEMPLATES_STORAGE_KEY, JSON.stringify(templates.slice(0, MAX_TEMPLATES)));
    return true;
  } catch {
    return false;
  }
}

/** ``text`` saved as ``name`` (replacing one of that name), first in the list. */
export function saveTemplate(name: string, text: string): QuestionTemplate[] {
  const cleanName = name.trim().slice(0, MAX_TEMPLATE_NAME_CHARS);
  const cleanText = text.trim().slice(0, MAX_QUESTION_CHARS);
  if (!cleanName || !cleanText) return readTemplates();
  const next = [
    { name: cleanName, text: cleanText },
    ...readTemplates().filter((t) => t.name.toLocaleLowerCase() !== cleanName.toLocaleLowerCase()),
  ];
  writeTemplates(next);
  return readTemplates();
}

export function deleteTemplate(name: string): QuestionTemplate[] {
  writeTemplates(readTemplates().filter((t) => t.name !== name));
  return readTemplates();
}

/** Where the first ``{placeholder}`` is in ``text`` (braces included), or null. */
export function firstPlaceholder(text: string): [number, number] | null {
  const match = /\{[^{}\n]{1,60}\}/.exec(text);
  return match ? [match.index, match.index + match[0].length] : null;
}
