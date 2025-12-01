https://app.datacamp.com/learn/skill-tracks/developing-large-language-models

# Create virtual environment

```
pyenv virtualenv 3.11.7 developing-large-language-models
echo "developing-large-language-models" > .python-version
```

# Install dependencies

Install pytorch following the instructions from https://pipenv.pypa.io/en/latest/indexes.html:
```
pip install pipenv
pipenv install numpy pandas
pipenv install torch torchvision torchaudio torchtext --index=pytorch
pipenv install torchmetrics
pipenv install matplotlib
pipenv install pillow
pipenv install nltk
pipenv install scikit-learn
pipenv install transformers sentencepiece
pipenv install datasets
pipenv install accelerate
pipenv install evaluate
pipenv install rouge_score
pipenv install trl
pipenv install llama-stack
pipenv install llama-cpp-python
```