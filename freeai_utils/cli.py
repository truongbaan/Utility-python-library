# freeai_utils/cli.py
import sys
import click
from ._setup import install_model

@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    if ctx.invoked_subcommand is None:
        # no subcommand given: show help and exit with error
        click.echo(ctx.get_help())
        sys.exit(1)

@main.command(help="Download default models for this library through hugging face ")
@click.argument("target", required=False)
def setup(target : str  = ""):
    install_model(target)
    
@main.command(help="Detail guide on command line")
def guide():
    print("*" * 100)
    print("HELP")
    print("*" * 100)
    print("setup: --A default models")
    print("       --S Speech To Text models")
    print("       --D Document related models")
    print("       --I Image OCR related models")
    print("       --T Translation models")
    print("       --L Default LLM models")
    print("       --ICF Image generator models")

if __name__ == "__main__":
    main()
