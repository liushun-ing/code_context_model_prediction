import os
from os.path import join

project_file_context = """<?xml version="1.0" encoding="UTF-8"?>
<projectDescription>
	<name>shunliu</name>
	<comment></comment>
	<projects>
	</projects>
	<buildSpec>
		<buildCommand>
			<name>org.eclipse.jdt.core.javabuilder</name>
			<arguments>
			</arguments>
		</buildCommand>
	</buildSpec>
	<natures>
		<nature>org.eclipse.jdt.core.javanature</nature>
	</natures>
</projectDescription>
"""


root_path = "D:/git_code/4000"

models = os.listdir(root_path)
models.sort(key=lambda x: int(x))

for model in models:
    print(model)
    model_path = join(root_path, model)
    repos = os.listdir(model_path)
    for repo in repos:
        project_path = join(model_path, repo, '.project')
        with open(project_path, 'w') as f:
            f.write(project_file_context.replace("shunliu", repo))
