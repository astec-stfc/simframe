from SimulationFramework.Framework_objects import frameworkElement


class drift(frameworkElement):
    """
    Class defining a drift object [deprecated?].
    """

    def __init__(self, *args, **kwargs):
        super(drift, self).__init__(
            *args,
            **kwargs,
        )

    # def _write_Elegant(self):
    #     wholestring=''
    #     etype = self._convertType_Elegant(self.objecttype)
    #     string = self.objectname+': '+ etype
    #     for key, value in list(merge_two_dicts(self.objectproperties, self.objectdefaults).items()):
    #         if not key is 'name' and not key is 'type' and not key is 'commandtype' and self._convertKeyword_Elegant(key) in elements_Elegant[etype]:
    #             value = getattr(self, key) if hasattr(self, key) and getattr(self, key) is not None else value
    #             key = self._convertKeyword_Elegant(key)
    #             value = 1 if value is True else value
    #             value = 0 if value is False else value
    #             tmpstring = ', '+key+' = '+str(value)
    #             if len(string+tmpstring) > 76:
    #                 wholestring+=string+',&\n'
    #                 string=''
    #                 string+=tmpstring[2::]
    #             else:
    #                 string+= tmpstring
    #     wholestring+=string+';\n'
    #     return wholestring
