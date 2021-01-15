from content import content_desc, content_metadata
from collabrative import collabrative

def hybrid(md,users,ls,title,userID,cr=None,kw=None,content_mode="metadata",collabrative_model=None,collabrative_mode="svd",filters={}):
	content_mode_opts = ['desc','metadata']
	assert content_mode in content_mode_opts, "Invalid content_mode. Choose from "+str(content_mode_opts)

	if content_mode=='metadata':
		assert cr is not None and kw is not None, "cr/kw parameters required when content_mode='metadata'"

	if content_mode=='metadata':
		md = content_metadata(title,md,ls,cr,kw)
	elif content_mode=='desc':
		md = content_desc(title,md,ls)

	res = collabrative(md,users,userID,model=collabrative_model,filters=filters,mode=collabrative_mode)

	return res