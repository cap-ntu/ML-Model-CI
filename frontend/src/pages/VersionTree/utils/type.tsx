export interface IGitData {
  author: {
    name: string;
    email: string;
  };
  hash: string;
  refs: string[];
  parents: string[];
  subject: string;
  created_at: string;
}