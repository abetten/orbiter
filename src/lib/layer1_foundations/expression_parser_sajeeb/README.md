## MacOS Setup
Follow this setup instructions to run the code in MacOS.

#### 1. Install homebrew
Copy and paste this in the terminal:
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> /Users/sajeeb/.zprofile
eval "$(/opt/homebrew/bin/brew shellenv)"
source ~/.zprofile
```

#### 2. Install bison
Copy and paste this in the terminal:
```bash
brew install bison
echo 'export PATH="/opt/homebrew/opt/bison/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

#### 3. Install flex
Copy and paste this in the terminal:
```bash
brew install flex
echo 'export PATH="/opt/homebrew/opt/flex/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

#### 4. Setup library paths for compiler
Copy and paste this in the terminal:
```bash
echo '
export LDFLAGS="-L/opt/homebrew/opt/flex/lib"
export CPPFLAGS="-I/opt/homebrew/opt/flex/include"
export LDFLAGS="-L/opt/homebrew/opt/bison/lib"
' >> ~/.zshrc
source ~/.zshrc
```
