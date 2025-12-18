# freeai_utils/cli.py
import sys
import click
from ._setup import install_library, install_model, remove_dir, check_for_updates
import os
from .cleaner import Cleaner
from .scripts import screenshot_ask_and_answer_clip

@click.group(invoke_without_command=True)
@click.pass_context
def main(ctx):
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())
        sys.exit(1)

@main.result_callback()
def check_update(result, **kwargs):
    """
    This function runs AFTER the subcommand to check for newer version of the lib.
    """
    check_for_updates("freeai-utils")

@main.command(help="Download default models for this library through hugging face ")
@click.argument("target", required=False)
@click.option("-y", "--yes", is_flag=True, help="Automatically confirm downloads without prompting")
def setup(target: str = "", yes: bool = False):
    install_model(target, yes)
    
@main.command(help="Detail guide on command line")
def help():
    print("*" * 100)
    print("HELP")
    print("*" * 100)
    # --- setup guide ---
    print("Usage: freeai-utils setup [FLAG]\n")
    print(" Flags:")
    print("   A   Default models                          →  freeai-utils setup A")
    print("   S   Speech-to-Text models                   →  freeai-utils setup S")
    print("   D   Document-related models                 →  freeai-utils setup D")
    print("   I   Image OCR models                        →  freeai-utils setup I")
    print("   T   Translation models                      →  freeai-utils setup T")
    print("   L   Default LLM models                      →  freeai-utils setup L")
    print("   ICF Image generator models                  →  freeai-utils setup ICF")
    print("   ICE Embeded for Image generator models      →  freeai-utils setup ICE")
    print("   V Vosk models      →  freeai-utils setup V")
    print("*" * 100)
    # --- clean guide ---
    print("Usage: freeai-utils clean [FLAG]\n")
    print("Description: Clean up downloaded files and model through setup command")
    print(" Flags:")
    print("   A   Remove both extra downloaded safetensors and vosk models      →  freeai-utils clean A")
    print("   ICF   Remove extra safetensors file downloaded from ICF setup     →  freeai-utils clean ICF")
    print("   V   Remove Vosk models                                            →  freeai-utils clean V")
    print("*" * 100)
    # --- secret-key guide ---
    print("Usage: freeai-utils secret-key [ACTION] [KEY|KEY=VALUE]\n")
    print("Description: Manage environment variables in the local .env file.")
    print(" Actions:")
    print("   add      Add or update a KEY=VALUE pair    →   freeai-utils secret-key add API_KEY=12345")
    print("   remove   Remove a key                      →   freeai-utils secret-key remove API_KEY")
    print("   read     Display all keys in .env          →   freeai-utils secret-key read")
    print("*" * 100)
    print("Usage: freeai-utils updates\n")
    print("Description: Checks for latest update on the library")
    print("*" * 100)
    # --- install-deps guide ---
    print("Usage: freeai-utils install-deps [FLAG]\n")
    print("Description: Install optional dependencies to reduce the initial library download size.")
    print(" Flags:")
    print("   ai     Install dependencies for AI features        →   freeai-utils install-deps ai")
    print("   all    Install all optional dependencies (default) →   freeai-utils install-deps all")
    print("*" * 100)
    
@main.command(help="Remove extra downloaded files")
@click.argument("target", required=False)
@click.option("-y", "--yes", is_flag=True, help="Automatically confirm delete without asking")
def clean(target : str = "", yes : bool = False):
    
    messages = {
        "V" : "Are you sure you want to remove only the Vosk model files? (Y/n): ",
        "ICF" : "Are you sure you want to remove only the safetensors files? (Y/n): ",
        "A" : "Are you sure you want to remove everything (Vosk + safetensors)? (Y/n): ",
        "" : "This will remove all downloaded files from civitai and the Vosk model.\n"
            "Use this before running pip uninstall.\n"
            "Are you sure you want to proceed? (Y/n): ",
    }
    
    key = target.strip().upper() if target is str else ""
    if key not in messages:
        print(f"Unknown target '{target}'. Valid options are: '', 'V', 'ICF', 'A'")
        return
    
    if not yes:
        decision = input(messages[key]).strip().lower()
            
        if decision != "y":
            print("Clean Up cancelled.")
            return
    
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    
    #known folder
    if key in ("A", ""):
        print("Cleaning: Vosk model and safetensors files...")
        known_folder = ["downloaded_models", "vosk_models"]
    elif key == "V":
        print("Cleaning: Vosk model files only...")
        known_folder = ["vosk_models"]
    elif key == "ICF":
        print("Cleaning: Extra safetensors files only...")
        known_folder = ["downloaded_models"]
    
    cleaner = Cleaner()
    cleaner.remove_all_files_end_with('.mp3') #for gttS audio
    cleaner.remove_all_files_end_with('.png')
    cleaner.remove_all_files_end_with('.env')
    
    for folder in known_folder:
        path = os.path.join(cur_dir, folder)
        try:
            remove_dir(path=path)
        except FileNotFoundError:
            pass
        except Exception as e:
            raise Exception(e)

@main.command(help="Checks for latest version")
def updates():
    result = check_for_updates("freeai-utils")
    if not result:
        click.echo("Library is up to dated")
        
@main.command(help="Built-in script to run gemini api, only for solving coding question.")
@click.argument("apikey", required=False)
@click.argument("hotkey", required=False, default="`")
def code_helper(apikey, hotkey):
    screenshot_ask_and_answer_clip(apikey, hotkey)

@main.command(help="Add secret key in the .env file")
@click.argument("action", required=True)
@click.argument("key_value_pair", required=False)
def secret_key(action: str, key_value_pair: str):
    """
    Updates or adds a key-value pair to a .env file located in the specified path.
    The key_value_pair must be in the format "KEY=VALUE".
    Args:
        path (str): The directory path where the .env file is located.
        key_value_pair (str): The configuration string (e.g., "API_KEY=12345").
    """
    
    action = action.lower() #normalize input
    if action not in ['add', 'remove', 'read']:
        print("Error: Invalid action. Must be 'add', 'remove', or 'read'.")
        return
    if action in ['add', 'remove'] and key_value_pair is None:
        print(f"Error: Action '{action}' requires a key or key=value pair.")
        return
    
    #init for holding value
    key = ""
    value = ""
    new_line = ""
    
    if key_value_pair is not None and '#' in key_value_pair:
        print("No comments or character '#' allowed!")
        return
    
    if action == 'add':
        # Validate the input format for 'add'
        if '=' not in key_value_pair or key_value_pair.count('=') > 1:
            print(f"Error: Invalid key-value pair format. Expected 'KEY=VALUE'. Received: {key_value_pair}")
            return
        
        # Extract key and value
        key, value = key_value_pair.split('=', 1)
        key = key.strip()
        value = value.strip()
        new_line = f"{key}={value}"
    
    elif action == 'remove':
        # INPUT KEY ONLY
        key = key_value_pair.strip()
        if not key or '=' in key:
            print(f"Error: For 'remove', only the key name should be provided (e.g., 'API_KEY'). Received: {key_value_pair}")
            return
    
    # file to save
    current = os.path.dirname(os.path.abspath(__file__))
    env_filepath = os.path.join(current, '.env')
    
    # Initialize a list to hold the lines of the environment file
    updated_lines = []
    key_found = False

    try:
        with open(env_filepath, 'r') as f:
            lines = f.readlines()
        print(f"Reading from existing .env file at {env_filepath}")
    except FileNotFoundError:
        lines = []
        if action == 'add':
            print(f"Note: .env file not found at {env_filepath}. A new one will be created.")
        elif action == 'remove':
            print(f"Error: .env file not found at {env_filepath}. Cannot remove key '{key}'.")
            return

    if action == 'read': 
        print(f"\n--- Content of .env file ---\n")
        for line in lines:
            print(line.strip())
        print("----------------------------------------------")
        return
    
    for line in lines:
        stripped_line = line.strip()

        if stripped_line and '=' in stripped_line:
            line_key = stripped_line.split('=', 1)[0].strip()
            if line_key == key:
                key_found = True
                if action == 'remove':
                    # skip this line to remove the key
                    print(f"Removed key `{key}` from .env.")
                    continue
                elif action == 'add':
                    # update key if exist
                    updated_lines.append(new_line + '\n')
                    print(f"Updated key `{key}` in .env.")
                    continue
                
        #preserve all other lines
        updated_lines.append(line)

    if not key_found:
        if action == 'add':
            # add
            if lines and not lines[-1].endswith('\n') and updated_lines:
                #ensure the last line ends with a newline before adding new content
                updated_lines.append('\n')
            updated_lines.append(new_line + '\n')
            print(f"Added new key `{key}` to .env.")
        elif action == 'remove':
            print(f"Warning: Key `{key}` was not found in .env. No action taken.")

    # write content back to the file
    try:
        if lines != updated_lines or (action == 'add' and not key_found):
            with open(env_filepath, 'w') as f:
                f.writelines(updated_lines)
            print(f"Successfully wrote updates to .env file")
        else:
             print("No changes were made to the .env file.")
             
    except IOError as e:
        print(f"Error writing to file {env_filepath}: {e}")

@main.command(help = "Install extra dependecies or ai lib that is heavy, this is made to reduce the intial download for the freeai_utils libraries")
@click.argument("value", required=False)
def install_deps(value: str) -> None: #install dependencies later for quick first init lib
    """
    Installs optional dependencies for freeai-utils using pip.
    Args:
        value (str): The name of the optional dependency group to install (e.g., 'ai').
    """
    
    # Validate the input value
    if value == "" or value is None:
        value = "all"
        
    VALID_GROUPS = ["ai", "all"]
    if value not in VALID_GROUPS:
        print(f"⚠️ Error: Invalid dependency group '{value}'.")
        print(f"   Available groups: {', '.join(VALID_GROUPS)}")
        return
    
    install_library(f'freeai_utils[{value}]')