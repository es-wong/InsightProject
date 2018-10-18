
def makeplot(df, plottype):
    names = list(df['feature'])
    scores = list(df['weight'])
    y = range(5,5-len(names),-1) # TODO(dieta) think about what to do here.
    fig, ax = plt.subplots(figsize=(2.5,4))
    plt.tick_params(top=False, bottom=False, left=False, right=False, labelleft=True, labelbottom=False)
    for spine in plt.gca().spines.values():
        spine.set_visible(False)
    colors = plt.cm.RdYlGn((np.sign(scores)+1)/2)
    if plottype == "good":
        ax.yaxis.set_ticks_position('right')
    plt.barh(y,scores,color=colors)
    plt.yticks(y, names, fontsize=16)
try:
    print("creating figure file")
    figfile = BytesIO()
    plt.savefig(figfile, format='svg', bbox_inches='tight')
    # Add plt.close() to avoid Assertion failed:
    # (NSViewIsCurrentlyBuildingLayerTreeForDisplay() != currentlyBuildingLayerTree)
    # errors, following https://stackoverflow.com/questions/49286741/matplotlib-not-working-with-python-2-7-and-django-on-osx
    plt.close()
    figfile.seek(0)
    figdata_svg = b'<svg' + figfile.getvalue().split(b'<svg')[1]
    return figdata_svg.decode('utf-8')
except err:
    print("exception", err)
    return None
