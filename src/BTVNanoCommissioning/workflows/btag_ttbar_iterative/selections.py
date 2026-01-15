from BTVNanoCommissioning.utils.selection import MET_filters, ele_mvatightid, jet_id, mu_idiso


def jet_selection(events, campaign: str):
    """Object selection for jets.

    :param events: event array
    :type events: NanoAOD array
    :param campaign: campaign name
    :type campaign: str
    :return: selected jets
    :rtype: NanoAOD array
    """
    mask = jet_id(events, campaign, max_eta=2.5, min_pt=20.0)  # jetIdTightLepVeto
    return events.Jet[mask]

def muon_selection(events, campaign: str):
    """Object selection for muons.

    :param events: event array
    :type events: NanoAOD array
    :param campaign: campaign name
    :type campaign: str
    :return: selected muons
    :rtype: NanoAOD array
    """
    muons = events.Muon
    mask = (muons.pt > 15) & (mu_idiso(events, campaign))
    return muons[mask]
def electron_selection(events, campaign: str):
    """Object selection for electrons.
    :param events: event array
    :type events: NanoAOD array
    :param campaign: campaign name
    :type campaign: str
    :return: selected electrons
    :rtype: NanoAOD array
    """
    electrons = events.Electron
    mask = (electrons.pt > 15) & (ele_mvatightid(events, campaign))
    return electrons[mask]


def met_selection(events, campaign):
    return MET_filters(events, campaign)
