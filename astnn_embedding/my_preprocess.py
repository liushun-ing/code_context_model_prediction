import os
from os.path import join

import javalang
import pandas as pd
from javalang.tree import FieldDeclaration

root_path = join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))), 'params_validation', 'git_repo_code',
                 'my_mylyn', 'repo_first_3', '1013')

# java_tsv = pd.read_csv(join(root_path, 'processed_java_codes.tsv'), delimiter='\t')
# print(java_tsv)

# ast_df = pd.read_pickle(join(root_path, 'astnn_ast.pkl'))
# dec: FieldDeclaration = ast_df.iloc[3]['code']
# print(dec.children)

train_vector_df = pd.read_pickle(join(root_path, 'mylyn_1_astnn_embedding.pkl'))
print(train_vector_df['embedding'][0].shape)


string = """
public abstract class AbstractRepositoryTaskEditor extends TaskFormPage {
	private static final String LABEL_HISTORY = "History";
	private TaskEditor parentEditor = null;
	private class TabVerifyKeyListener implements VerifyKeyListener {
		public void verifyKey(VerifyEvent event) {  }
	}
	protected void initTaskEditor(IEditorSite site, RepositoryTaskEditorInput input) {  }
	protected Label createLabel(Composite composite, RepositoryTaskAttribute attribute) {  }
	private void setText(Browser browser, String html) {}
	private class RadioButtonListener implements SelectionListener, ModifyListener {
		public void widgetDefaultSelected(SelectionEvent e) {}
		public void modifyText(ModifyEvent e) {  }
	}
}

"""
tokens = javalang.tokenizer.tokenize(string)
parser = javalang.parser.Parser(tokens)
tree = parser.parse_class_or_interface_declaration()
print(tree.attrs)
print(tree)
print(tree.children)
print(tree.children[4][0].children[4][0].children)
