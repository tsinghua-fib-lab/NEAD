import argparse

def add_minus_flags(parser: argparse.ArgumentParser):
    """
    Add hyphenated aliases for argparse options that contain underscores.
    For example, --fix_existing receives the alias --fix-existing.
    """
    new_option_string_actions = {}
    for group in parser._action_groups:
        for action in group._group_actions:
            # Only process optional arguments.
            if not action.option_strings:
                continue
            # Use the longest double-dash option as the canonical spelling.
            original_option = ''
            for opt in list(action.option_strings):
                if opt.startswith('--') and '_' in opt and len(opt) > len(original_option):
                    original_option = opt
            # Create an alias when a suitable option was found.
            if original_option:
                # Replace underscores with hyphens.
                new_alias_option = original_option.replace('_', '-')
                # Add the alias to the action.
                if new_alias_option not in action.option_strings:
                    action.option_strings.append(new_alias_option)
            # Rebuild the lookup table with both spellings.
            for opt in action.option_strings:
                new_option_string_actions[opt] = action
    # Replace argparse's internal option lookup table.
    parser._option_string_actions = new_option_string_actions
    return parser

def add_negation_flags(parser: argparse.ArgumentParser):
    """
    Add a matching --no-xxx option for each store_true argument.
    """
    for action in parser._actions:
        # Only process boolean store_true actions.
        if isinstance(action, argparse._StoreTrueAction):
            # Obtain the option name, for example --flag.
            for option in action.option_strings:
                if not option.startswith('--'): 
                    continue
                neg_option = '--no-' + option.removeprefix('--')
                if any(neg_option in a.option_strings for a in parser._actions):
                    continue # Avoid duplicate aliases.
                parser.add_argument(
                    neg_option,
                    dest=action.dest,
                    action='store_false',
                    default=action.default,
                    help=f"Disable {option.removeprefix('--')}"
                )
    return parser
