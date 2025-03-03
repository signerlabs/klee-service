# coding:utf8
import subprocess

def convert_keynote_to_pdf(keynote_path:str, pdf_path:str):
    """
    AppleScript command to open the .key file and export it as a PDF
    """
    applescript = f'''
    tell application "Keynote"
        open POSIX file "{keynote_path}"
        set theDoc to the front document
        set theDocName to name of theDoc
        set thePdfPath to POSIX file "{pdf_path}"
        export theDoc to thePdfPath as PDF
        close theDoc
    end tell
    '''

    # Run the AppleScript command
    subprocess.run(['osascript', '-e', applescript])
