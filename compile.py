import stickytape

def compile(path : str):
    output = stickytape.script(
        path = path, 
        add_python_paths=["./src"], 
    )
    # make the output to be /build/[path]
    # remove .py but not the file name of the path
    path = path.split("/")[-1].replace(".py", "")
    output_path = f"./build/{path}_standalone.py"
    with open(output_path, "w") as f:
        f.write(output)

if __name__ == "__main__":
    compile("./main.py")
    compile("./limelight.py")