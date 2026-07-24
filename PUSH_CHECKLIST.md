# GitHub Push Checklist

Before the first commit:

- [ ] The repository contains only the two phishing project folders.
- [ ] `.venv` is not staged.
- [ ] `model.pkl` is not staged.
- [ ] `lstm_model.pt` is not staged.
- [ ] Raw datasets are not staged unless redistribution is explicitly permitted.
- [ ] No passwords, tokens, API keys, or private credentials are present.
- [ ] Both confusion-matrix images are present.
- [ ] The LSTM training-history image is present.
- [ ] `git status` has been reviewed before committing.
- [ ] The GitHub repository was created empty, without a server-side README, license, or `.gitignore`.

Useful checks:

```powershell
git status
git diff --cached --stat
git remote -v
git branch --show-current
```
