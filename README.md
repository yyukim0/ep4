### Como rodar

```bash
# Salve o link do repositório e ponha no terminal
git clone (link do repositório)
```

```bash
# Para instalar as dependências no terminal
pip install -r requirements.txt
```


Antes de começar a implementar, rode:

```bash
# Valida todas as pistas
python tests/validar_pistas.py

# Testa o ambiente com agente trivial (acelera 3x e segue reto)
PYTHONPATH=src python src/env.py pistas/pista_01.txt

# Anima um episódio no terminal
PYTHONPATH=src python src/visualize.py pistas/pista_01.txt
```

Se todas as três rodarem sem erro, o ambiente está pronto.
