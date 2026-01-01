class DictEncoder():
    def encodeObject(self, object):
        if object is None or isinstance(object, (int, bool, str,  float)):
            return object
        elif isinstance(object, (list, tuple)):
            return [self.encodeObject(i) for i in object]
        elif isinstance(object, (dict)):
            return {key: self.encodeObject(value) for key, value in object.items()}
        else:
            try:
                return {key: self.encodeObject(value) for key, value in vars(object).items()}
            except TypeError:
                return str(object)
    
    def encodeTypes(self, typesList):
        typesDict = {}
        for singleType in typesList:
            typesDict[singleType.__qualname__] = singleType

        return typesDict