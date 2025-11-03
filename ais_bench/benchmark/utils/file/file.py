from typing import List, Tuple, Union
import os
import json
import fnmatch
import tabulate

from ais_bench.benchmark.utils.logging.logger import AISLogger
from ais_bench.benchmark.utils.logging.exceptions import FileMatchError
from ais_bench.benchmark.utils.logging.error_codes import UTILS_CODES

logger = AISLogger()

def write_status(file_path, status):
    """Write status to a JSON file, appending to existing content.
    
    Args:
        file_path: Path to the status file
        status: Status data to append
        
    Returns:
        bool: True if successful, False otherwise
    """
    # read existing content
    existing_data = []
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse JSON from status file '{file_path}': {e}. "
                f"Starting with empty status list."
            )
            existing_data = []
        except IOError as e:
            logger.warning(
                f"Failed to read status file '{file_path}': {e}. "
                f"Starting with empty status list."
            )
            existing_data = []

    # add new status
    existing_data.append(status)

    # write to file
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f)
        logger.debug(f"Successfully wrote status to '{file_path}' ({len(existing_data)} total statuses)")
        return True
    except IOError as e:
        logger.warning(f"Failed to write status to '{file_path}': {e}")
        return False


def read_and_clear_statuses(tmp_file_dir, tmp_file_name_list):
    """Read all task statuses and clear temporary files.
    
    Args:
        tmp_file_dir: Directory containing temporary status files
        tmp_file_name_list: List of temporary file names to process
        
    Returns:
        list: List of all task statuses, empty list if error occurs
    """
    if not os.path.exists(tmp_file_dir):
        logger.debug(f"Temporary file directory does not exist: '{tmp_file_dir}'")
        return []

    abs_path_list = [os.path.join(tmp_file_dir, tmp_file_name) for tmp_file_name in tmp_file_name_list]
    all_status = []
    
    logger.debug(f"Reading and clearing {len(abs_path_list)} status files from '{tmp_file_dir}'")
    
    for tmp_file in abs_path_list:
        try:
            # read existing content
            with open(tmp_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            status_count = len(data)
            all_status.extend(data)
            
            # clear file content
            with open(tmp_file, 'w', encoding='utf-8') as f:
                json.dump([], f)
            
            logger.debug(f"Read {status_count} statuses from '{tmp_file}' and cleared file")

        except json.JSONDecodeError as e:
            logger.warning(
                f"Failed to parse JSON from '{tmp_file}': {e}. "
                f"Clearing corrupted file and continuing."
            )
            try:
                with open(tmp_file, 'w', encoding='utf-8') as f:
                    json.dump([], f)
            except IOError as write_err:
                logger.warning(f"Failed to clear corrupted file '{tmp_file}': {write_err}")
                
        except IOError as e:
            logger.warning(
                f"Failed to read status file '{tmp_file}': {e}. "
                f"Attempting to clear and continuing."
            )
            try:
                with open(tmp_file, 'w', encoding='utf-8') as f:
                    json.dump([], f)
            except IOError as write_err:
                logger.warning(f"Failed to clear unreadable file '{tmp_file}': {write_err}")
    
    logger.debug(f"Total statuses collected: {len(all_status)}")
    return all_status


def match_files(path: str,
                pattern: Union[str, List],
                fuzzy: bool = False) -> List[Tuple[str, str]]:
    """Match files in directory based on pattern(s).
    
    Args:
        path: Directory path to search
        pattern: Single pattern string or list of patterns to match
        fuzzy: If True, wraps patterns with wildcards (*pattern*)
        
    Returns:
        List of tuples (filename_without_ext, full_path), sorted by filename
    """
    if isinstance(pattern, str):
        pattern = [pattern]
    if fuzzy:
        pattern = [f'*{p}*' for p in pattern]
    
    logger.debug(f"Searching for files in '{path}' with patterns: {pattern} (fuzzy={fuzzy})")
    
    files_list = []
    for root, _, files in os.walk(path):
        for name in files:
            for p in pattern:
                if fnmatch.fnmatch(name.lower(), p.lower()):
                    files_list.append((name[:-3], os.path.join(root, name)))
                    break

    logger.debug(f"Found {len(files_list)} matching files")
    return sorted(files_list, key=lambda x: x[0])


def match_cfg_file(workdir: Union[str, List[str]],
                   pattern: Union[str, List[str]]) -> List[Tuple[str, str]]:
    """Match config files in workdir recursively given the pattern.

    Additionally, if the pattern itself points to an existing file, it will be
    directly returned.
    
    Args:
        workdir: Single directory path or list of directory paths to search
        pattern: Single pattern string or list of patterns to match (auto-appends .py)
        
    Returns:
        List of tuples (filename_without_ext, full_path) for matched config files
        
    Raises:
        FileMatchError: If patterns match 0 or more than 1 config file
    """
    def _mf_with_multi_workdirs(workdir, pattern, fuzzy=False):
        if isinstance(workdir, str):
            workdir = [workdir]
        files = []
        for wd in workdir:
            files += match_files(wd, pattern, fuzzy=fuzzy)
        return files

    if isinstance(pattern, str):
        pattern = [pattern]
    pattern = [p + '.py' if not p.endswith('.py') else p for p in pattern]
    
    logger.debug(f"Matching config files in {workdir} with patterns: {[p[:-3] for p in pattern]}")
    
    files = _mf_with_multi_workdirs(workdir, pattern, fuzzy=False)
    if len(files) != len(pattern):
        nomatched = []
        ambiguous = []
        ambiguous_return_list = []
        err_msg = ("The provided pattern matches 0 or more than one "
                   "config. Please verify your pattern and try again. \n")
        for p in pattern:
            files_ = _mf_with_multi_workdirs(workdir, p, fuzzy=False)
            if len(files_) == 0:
                nomatched.append([p[:-3]])
                logger.debug(f"No matches found for pattern: {p[:-3]}")
            elif len(files_) > 1:
                ambiguous.append([p[:-3], '\n'.join(f[1] for f in files_)])
                ambiguous_return_list.append(files_[0])
                logger.debug(f"Multiple matches found for pattern '{p[:-3]}': {len(files_)} files")
        if nomatched:
            table = [['Not matched patterns'], *nomatched]
            err_msg += tabulate.tabulate(table,
                                         headers='firstrow',
                                         tablefmt='psql')
            err_msg += "\n"
            logger.debug(f"Failed to match {len(nomatched)} pattern(s): {[p[0] for p in nomatched]}")
        if ambiguous:
            table = [['Ambiguous patterns', 'Matched files'], *ambiguous]
            warning_msg = 'Found ambiguous patterns, using the first matched config.\n'
            warning_msg += tabulate.tabulate(table,
                                         headers='firstrow',
                                         tablefmt='psql')
            logger.warning(warning_msg)
            return ambiguous_return_list

        raise FileMatchError(UTILS_CODES.MATCH_CONFIG_FILE_FAILED, err_msg)
    return files

def search_configs_from_args(args):
    """Search and collect configuration files based on command line arguments.
    
    Args:
        args: Command line arguments containing models, datasets, and summarizer
        
    Returns:
        Prints a formatted table of found configuration files
    """
    logger.info('Searching for configuration files...')
    table = [["Task Type", "Task Name", "Config File Path"]]
    
    # parse model args
    if args.models:
        logger.debug(f"Processing {len(args.models)} model argument(s): {args.models}")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        default_configs_dir = os.path.join(parent_dir, 'configs')
        models_dir = [
            os.path.join(args.config_dir, 'models'),
            os.path.join(default_configs_dir, './models'),
        ]
        for model_arg in args.models:
            for model in match_cfg_file(models_dir, [model_arg]):
                table.append(["--models", model[0], os.path.abspath(model[1])])
                logger.debug(f"Found model config: {model[0]}")

    # parse dataset args
    if args.datasets:
        logger.debug(f"Processing {len(args.datasets)} dataset argument(s): {args.datasets}")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        default_configs_dir = os.path.join(parent_dir, 'configs')
        datasets_dir = [
            os.path.join(args.config_dir, 'datasets'),
            os.path.join(args.config_dir, 'dataset_collections'),
            os.path.join(default_configs_dir, './datasets'),
            os.path.join(default_configs_dir, './dataset_collections')
        ]
        for dataset_arg in args.datasets:
            if '/' in dataset_arg:
                dataset_name, _ = dataset_arg.split('/', 1)
                logger.debug(f"Dataset argument contains '/': using '{dataset_name}' from '{dataset_arg}'")
            else:
                dataset_name = dataset_arg

            for dataset in match_cfg_file(datasets_dir, [dataset_name]):
                table.append(["--datasets", dataset[0], os.path.abspath(dataset[1])])
                logger.debug(f"Found dataset config: {dataset[0]}")

    # parse summarizer args
    if args.summarizer:
        summarizer_arg = args.summarizer if args.summarizer is not None else 'example'
        logger.debug(f"Processing summarizer argument: {summarizer_arg}")
        script_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(script_dir)
        default_configs_dir = os.path.join(parent_dir, 'configs')
        summarizers_dir = [
            os.path.join(args.config_dir, 'summarizers'),
            os.path.join(default_configs_dir, './summarizers'),
        ]

        # Check if summarizer_arg contains '/'
        if '/' in summarizer_arg:
            # If it contains '/', split the string by '/'
            # and use the second part as the configuration key
            summarizer_file, _ = summarizer_arg.split('/', 1)
            logger.debug(f"Summarizer argument contains '/': using '{summarizer_file}' from '{summarizer_arg}'")
        else:
            # If it does not contain '/', keep the original logic unchanged
            summarizer_file = summarizer_arg

        s = match_cfg_file(summarizers_dir, [summarizer_file])[0]
        table.append(["--summarizer", s[0], os.path.abspath(s[1])])
        logger.debug(f"Found summarizer config: {s[0]}")
    
    logger.info(f"Configuration search completed. Found {len(table) - 1} config file(s).")
    print(
        tabulate.tabulate(
            table,
            headers='firstrow',
            tablefmt="fancy_grid",
            stralign="left",
            missingval="N/A",
        )
    )

def check_mm_custom(path):
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = json.loads(line.strip())
            if "type" not in line or "path" not in line:
                return False
            elif line["type"] not in ["image", "video", "audio"]:
                return False
    return True
