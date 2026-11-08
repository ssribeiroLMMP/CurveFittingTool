# Curve Fitting Application

Aplicação desktop de ajuste de curvas em Python/Tkinter, com núcleo científico
orientado a objetos. O projeto está em desenvolvimento incremental; ainda não há
um ponto de entrada executável para a interface gráfica.

## Ambiente de desenvolvimento com uv

O desenvolvimento usa Python compatível com `requires-python` no `pyproject.toml`
e um ambiente virtual local `.venv`, gerenciado
pelo [uv](https://docs.astral.sh/uv/). Docker não é necessário.

### Instalação inicial no Windows (PowerShell)

Instale o uv uma vez com o
[instalador oficial](https://docs.astral.sh/uv/getting-started/installation/):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Abra um novo terminal para disponibilizar `uv` no `PATH`. Na raiz do repositório:

```powershell
uv sync --locked
uv run --locked pytest
```

O uv seleciona um Python compatível com `requires-python` no `pyproject.toml`,
baixa um interpretador se necessário e cria `.venv`,
instala o pacote `curvefit` em modo editável e inclui o grupo `dev` com pytest.
Alterações em `src/` ficam disponíveis sem reinstalação manual e sem configurar
`PYTHONPATH`. Não é necessário ativar o ambiente para usar `uv run`.

Em Linux/macOS, instale o uv conforme a documentação oficial e use os mesmos
comandos `uv sync` e `uv run` na raiz do repositório.

### Comandos do dia a dia

```powershell
# Sincronizar exatamente as dependências versionadas
uv sync --locked

# Executar todos os testes ou um pacote específico
uv run --locked pytest
uv run --locked pytest tests/unit/domain/parameter

# Abrir o interpretador do projeto
uv run --locked python

# Verificar a instalação do Tkinter sem abrir uma janela
uv run --locked python -c "import tkinter; print(tkinter.TkVersion)"

# Abrir a janela de demonstração do Tkinter (feche-a para continuar)
uv run --locked python -m tkinter
```

No editor, selecione `.venv/Scripts/python.exe` no Windows ou `.venv/bin/python`
no Linux/macOS como interpretador do projeto.

### Dependências e reprodutibilidade

`pyproject.toml` declara as dependências científicas alinhadas à baseline v1.5:
NumPy, SymPy, SciPy, LMFIT, Pandas e Matplotlib, além de Lark para parsing e
openpyxl para leitura de XLSX. A presença de uma
biblioteca no ambiente não altera as restrições de importação de cada camada.
pytest fica separado no grupo de desenvolvimento `dev`.

`uv.lock` registra as versões resolvidas e deve ser versionado, assim como
`pyproject.toml`. `.venv`, caches e saídas geradas ficam fora
do Git. Após clonar ou atualizar o repositório, use `uv sync --locked`.

```powershell
# Adicionar uma dependência de execução ou de desenvolvimento
uv add nome-do-pacote
uv add --dev nome-da-ferramenta

# Atualizar intencionalmente as versões e validar o resultado
uv lock --upgrade
uv sync --locked
uv run --locked pytest
```

Revise e versione as alterações de `pyproject.toml` e `uv.lock` juntas quando
ambos forem modificados.

Tkinter faz parte da distribuição do Python e não deve ser instalado via pip/uv.
Se o import falhar, use uma distribuição Python com Tcl/Tk; em Debian/Ubuntu,
instale também o pacote de sistema `python3-tk` correspondente ao interpretador.
Para testar a GUI é necessário um ambiente gráfico; os testes do núcleo devem
continuar funcionando sem abrir janelas.

## Referência arquitetural

A baseline normativa indicada em `AGENTS.md` é:

`docs/Curve_Fitting_Application_Architecture_Baseline_v1.5.pdf`

A v1.5 substitui as versões anteriores; documentos em `docs/old/` são históricos.

As instruções de desenvolvimento estão em `AGENTS.md`.

## Organização do repositório

- `src/curvefit/`: pacotes de produção.
- `tests/unit/`: testes unitários espelhando os pacotes de produção.
- `tests/integration/`: testes entre módulos e de fluxos completos.
- `tests/fixtures/`: dados sintéticos e não sensíveis para testes.
- `docs/`: documentação do projeto.
- `examples/`: exemplos de entrada não sensíveis.
- `scripts/`: scripts auxiliares de desenvolvimento e manutenção.
