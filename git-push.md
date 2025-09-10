# How to Push This Project to GitHub

These instructions assume you already have a GitHub repository created and want to push your local project to it.

## 1. Initialize Git (if not already done)

```bash
git init
```

## 2. Add GitHub as Remote

Replace the URL with your actual GitHub repo:

```bash
git remote add origin https://github.com/VimalVasu/rag-agent.git
```

To confirm:

```bash
git remote -v
```

## 3. Stage and Commit Changes

```bash
git add .
git commit -m "Initial commit"
```

## 4. Push to GitHub

If it's your first push:

```bash
git branch -M main
```

Then push:

```bash
git push -u origin main
```

If you need to overwrite existing remote files (e.g., README):

```bash
git push -u origin main --force
```

## 5. Future Commits

For any updates:

```bash
git add .
git commit -m "Your message here"
git push
```

---

## Notes

* Use `git status` to check which files have changed
* Use `git log` to see commit history
* Always commit meaningful, grouped changes
