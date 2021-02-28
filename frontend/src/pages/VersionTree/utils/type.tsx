export interface IGitData {
  author: {
    name: string | null;
    email: string | null;
  };
  hash: string;
  refs: string[];
  parents: string[];
  subject: string | null;
  created_at: string;
}