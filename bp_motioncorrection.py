"""Create and submit SLURM job script to motion correct set of TIFFs.
"""
import argparse
import logging
import os
import subprocess
import sys

logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler(stream=sys.stdout))
logger.setLevel(logging.DEBUG)

TEMPLATE_SCRIPT = """
#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=10
#SBATCH --time=5:00:00
#SBATCH --mem=60GB
#SBATCH --job-name=motionCorrectionTest
#SBATCH --mail-type=END
#SBATCH --mail-user={email}
#SBATCH --output=slurm_%j.out
#SBATCH --array=0-{numTIFFsMinusOne}
# Expects two input parameters:
#   $1: Working directory containing tifs to correct
#   $2: File name pattern to select tifs, i.e. Run0034_00*.tif.
#        If omitted, uses all .tif that are not Ref.tif.

module purge
module load matlab/R2018a

cd {workingDir}

{extglob}
tifs=(`pwd`/{pattern})

tif=${{tifs[$SLURM_ARRAY_TASK_ID]}}

{{
    echo $tif
    matlab -nodisplay -r "normcorremotioncorrection_single('$tif','Ref.tif'); exit"
}} > $tif.log 2>&1

exit
"""


def run_shell_command(scommand, capture_output=False):
    """[summary]

    Arguments:
        scommand {[type]} -- [description]

    Keyword Arguments:
        capture_output {bool} -- [description] (default: {False})

    Raises:
        RuntimeError: [description]

    Returns:
        [type] -- [description]
    """
    logger.debug("About to run:\n{}".format(scommand))
    if capture_output:
        process_results = subprocess.run(scommand, shell=True, stdout=subprocess.PIPE)
    else:
        process_results = subprocess.run(scommand, shell=True)
    if process_results.returncode:
        raise RuntimeError("Received non-zero return code: {}".format(process_results))
    return process_results


def fill_in_template(template, params):
    return template.format(**params)


def make_submit_script(
    params, fname="submit_motioncorrection.sh", template=TEMPLATE_SCRIPT
):
    # check if we need to turn extended glob option on
    if "!" in args.pattern:
        params["extendedGlob"] = "shopt -s extglob\n"
    else:
        params["extendedGlob"] = ""
    # get number of TIFs Minus One -- specifying last index in a zero-index array
    scommand = f"ls {os.path.join(params['workingDir'], '')}{params['pattern']} | wc -l"
    ls_result = run_shell_command(scommand, capture_output=True)
    params["numTIFFsMinusOne"] = int(ls_result.stdout) - 1

    script = fill_in_template(template, params)
    with open(fname, "w") as scriptFile:
        scriptFile.write(script)
    return fname


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--workingDir", default=".", help="Working directory containing tifs to correct"
    )
    parser.add_argument(
        "--pattern",
        # Default will include all tif EXCEPT Ref.tif
        default="!(Ref).tif",
        help="File name pattern to select tifs, i.e. Run0034_00*.tif.",
    )
    try:
        parser.add_argument(
            "--email",
            default=f'{os.getenv("USER")}@nyu.edu',
            help="Notification email address.",
        )
    except OSError:
        parser.add_argument(
            "--email",
            required=True,
            help="Notification email address. Entry required on your system.",
        )

    args = parser.parse_args()
    scriptFilename = make_submit_script(vars(args))
